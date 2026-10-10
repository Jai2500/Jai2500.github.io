// Hand-written WebGPU runtime for the conv U-Net world model (tinywm/unet.py).
// Same interface as GpuDiT: reset(), step(x, t, action, actMask, wantOutput),
// commit(), plus denoise(noise, ts, action), which runs every Euler step of a
// frame on the GPU with a single readback. The context is the input of the last
// committed call (the previous frame re-noised at sigma_ctx).
//
// Kernels:
//   CONV   tiled direct convolution: a workgroup computes an 8x8 pixel tile for
//          8 output channels; input patches and weights are staged in workgroup
//          memory 8 input channels at a time. GroupNorm + FiLM + SiLU of the input
//          is fused into the patch load (per-channel scale/shift from STATS), and
//          the residual add into the epilogue.
//   STATS  GroupNorm statistics -> per-channel (a, b) with silu(x * a + b) the
//          normalized, FiLM-modulated activation.
//   CONVT  2x2 stride-2 transposed convolution.   COPY  concat helper.
//   EULER  x += dt * v (clamped to [-1, 1] after the last step).

import { parseWeights } from './weights.js';

export { parseWeights };

function matvec(W, b, x, nOut, nIn) {
  const y = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = b[o];
    for (let i = 0, r = o * nIn; i < nIn; i++) s += W[r + i] * x[i];
    y[o] = s;
  }
  return y;
}
const silu = (v) => v.map((x) => x / (1 + Math.exp(-x)));

const CONV = /* wgsl */ `
struct P { cin: u32, cout: u32, hin: u32, win: u32, hout: u32, wout: u32, k: u32, stride: u32,
           pad: u32, act: u32, res: u32, accum: u32 };
const TS: u32 = 8u;   // output tile side
const CO: u32 = 8u;   // output channels per thread
const CI: u32 = 8u;   // input channels per chunk
const PM: u32 = 17u;  // max patch side: (TS - 1) * 2 + 3
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> Bv: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> AB: array<f32>;
@group(0) @binding(5) var<storage, read> R: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> tile: array<f32, 2312>;  // CI * PM * PM
var<workgroup> wts: array<f32, 576>;     // CO * CI * 9
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let ox = wg.x * TS + l.x; let oy = wg.y * TS + l.y; let co0 = wg.z * CO;
  let ps = (TS - 1u) * p.stride + p.k; let kk = p.k * p.k;
  let ix0 = i32(wg.x * TS * p.stride) - i32(p.pad); let iy0 = i32(wg.y * TS * p.stride) - i32(p.pad);
  var acc: array<f32, 8>;
  for (var c0 = 0u; c0 < p.cin; c0 += CI) {
    for (var i = li; i < CI * ps * ps; i += 64u) {
      let ci = i / (ps * ps); let r = i % (ps * ps); let py = r / ps; let px = r % ps;
      let c = c0 + ci; let iy = iy0 + i32(py); let ix = ix0 + i32(px);
      var v = 0.0;
      if (c < p.cin && iy >= 0 && iy < i32(p.hin) && ix >= 0 && ix < i32(p.win)) {
        v = X[(c * p.hin + u32(iy)) * p.win + u32(ix)];
        if (p.act == 1u) { let z = v * AB[2u * c] + AB[2u * c + 1u]; v = z / (1.0 + exp(-z)); }
      }
      tile[(ci * PM + py) * PM + px] = v;
    }
    for (var i = li; i < CO * CI * kk; i += 64u) {
      let co = i / (CI * kk); let r = i % (CI * kk); let ci = r / kk; let q = r % kk;
      var w = 0.0;
      if (co0 + co < p.cout && c0 + ci < p.cin) { w = Wt[((co0 + co) * p.cin + c0 + ci) * kk + q]; }
      wts[(co * CI + ci) * 9u + q] = w;
    }
    workgroupBarrier();
    for (var ci = 0u; ci < CI; ci++) {
      for (var ky = 0u; ky < p.k; ky++) {
        for (var kx = 0u; kx < p.k; kx++) {
          let v = tile[(ci * PM + l.y * p.stride + ky) * PM + l.x * p.stride + kx];
          let wb = ci * 9u + ky * p.k + kx;
          for (var co = 0u; co < CO; co++) { acc[co] += wts[co * CI * 9u + wb] * v; }
        }
      }
    }
    workgroupBarrier();
  }
  if (ox >= p.wout || oy >= p.hout) { return; }
  for (var co = 0u; co < CO; co++) {
    let c = co0 + co;
    if (c >= p.cout) { break; }
    let o = (c * p.hout + oy) * p.wout + ox;
    var y = acc[co] + Bv[c];
    if (p.res == 1u) { y += R[o]; }
    if (p.accum == 1u) { y += Y[o]; }
    Y[o] = y;
  }
}`;

const STATS = /* wgsl */ `
struct P { c: u32, hw: u32, groups: u32, film: u32, fs: u32, fh: u32, a: u32, b: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> AB: array<f32>;
@group(0) @binding(2) var<storage, read> Gm: array<f32>;
@group(0) @binding(3) var<storage, read> Bt: array<f32>;
@group(0) @binding(4) var<storage, read> C: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let cpg = p.c / p.groups; let n = cpg * p.hw; let base = wg.x * n;
  var a = 0.0;
  for (var i = l.x; i < n; i += 256u) { a += X[base + i]; }
  red[l.x] = a;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { red[l.x] += red[l.x + s]; } workgroupBarrier(); }
  let mean = red[0] / f32(n);
  workgroupBarrier();
  var b = 0.0;
  for (var i = l.x; i < n; i += 256u) { let d = X[base + i] - mean; b += d * d; }
  red[l.x] = b;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { red[l.x] += red[l.x + s]; } workgroupBarrier(); }
  let rstd = inverseSqrt(red[0] / f32(n) + 1e-5);
  if (l.x < cpg) {
    let c = wg.x * cpg + l.x;
    var sa = rstd * Gm[c];
    var sb = Bt[c] - mean * sa;
    if (p.film == 1u) { let f = 1.0 + C[p.fs + c]; sa *= f; sb = sb * f + C[p.fh + c]; }
    AB[2u * c] = sa; AB[2u * c + 1u] = sb;
  }
}`;

const CONVT = /* wgsl */ `
struct P { cin: u32, cout: u32, hin: u32, win: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> Bv: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let wo = 2u * p.win; let hw = 4u * p.hin * p.win;
  if (g.x >= p.cout * hw) { return; }
  let co = g.x / hw; let r = g.x % hw; let oy = r / wo; let ox = r % wo;
  let iy = oy / 2u; let ix = ox / 2u; let ky = oy % 2u; let kx = ox % 2u;
  var acc = Bv[co];
  for (var ci = 0u; ci < p.cin; ci++) {
    acc += X[(ci * p.hin + iy) * p.win + ix] * Wt[((ci * p.cout + co) * 2u + ky) * 2u + kx];
  }
  Y[g.x] = acc;
}`;

const COPY = /* wgsl */ `
struct P { n: u32, off: u32, a: u32, b: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x < p.n) { Y[p.off + g.x] = X[g.x]; }
}`;

const EULER = /* wgsl */ `
struct P { dt: f32, last: u32, a: u32, b: u32 };
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= 12288u) { return; }
  var x = X[g.x] + p.dt * V[g.x];
  if (p.last == 1u) { x = clamp(x, -1.0, 1.0); }
  X[g.x] = x;
}`;

const MAX_NFE = 8;

export class GpuUNet {
  static async create(weights, opts = {}) {
    if (!navigator.gpu) throw new Error('no WebGPU');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no adapter');
    if (adapter.info?.isFallbackAdapter && !opts.allowFallback) throw new Error('fallback adapter');
    return new GpuUNet(await adapter.requestDevice(), weights);
  }

  constructor(device, w) {
    const cfg = w.cfg, ch = cfg.chans, cdim = cfg.cdim, R = 64, NPIX = 3 * R * R;
    if (ch.length !== 3 || cfg.in_ch !== 3 || cfg.ctx_frames !== 1) throw new Error('unsupported config');
    this.device = device;
    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    const buf = (n) => device.createBuffer({ size: Math.max(16, n * 4), usage: S | CD | CS });
    const upload = (arr) => {
      const b = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: S, mappedAtCreation: true });
      new Float32Array(b.getMappedRange(0, arr.byteLength)).set(arr);
      b.unmap();
      return b;
    };
    const uni = (arr) => {
      const b = device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.UNIFORM | CD, mappedAtCreation: true });
      new Uint32Array(b.getMappedRange()).set(arr);
      b.unmap();
      return b;
    };
    const cache = new Map();
    const W = (n) => cache.get(n) || cache.set(n, upload(w.get(n))).get(n);
    const pipe = (code) => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    const P = (this.P = { conv: pipe(CONV), stats: pipe(STATS), convt: pipe(CONVT), copy: pipe(COPY), euler: pipe(EULER) });
    const dummy = buf(4);
    const wg = (n) => Math.ceil(n / 64);

    // CPU conditioning: timestep/action MLPs, then one FiLM vector per residual block.
    this.cpu = {};
    for (const k of ['t_mlp.0', 't_mlp.2', 'a_mlp.0', 'a_mlp.2']) this.cpu[k] = [w.get(k + '.weight'), w.get(k + '.bias')];
    this.cpu.null = w.get('null_action');
    this.cdim = cdim;
    this.films = []; // [weight, bias, cout, offset]
    let filmLen = 0;

    // The plan: [pipeline, (condBuffer) => bind-group buffers, [x, y, z]], materialized per cond buffer.
    const plan = [];
    const conv = (X, name, Y, cin, cout, hin, k, stride, pad, { act = null, res = null, accum = false } = {}) => {
      const hout = Math.floor((hin + 2 * pad - k) / stride) + 1;
      const u = uni(new Uint32Array([cin, cout, hin, hin, hout, hout, k, stride, pad, act ? 1 : 0, res ? 1 : 0, accum ? 1 : 0]));
      const Wt = W(name + '.weight'), Bv = W(name + '.bias');
      plan.push([P.conv, () => [X, Wt, Bv, Y, act || dummy, res || dummy, u], [Math.ceil(hout / 8), Math.ceil(hout / 8), Math.ceil(cout / 8)]]);
    };
    const stats = (X, AB, name, c, res, groups, film = null) => {
      const u = uni(new Uint32Array([c, res * res, groups, film === null ? 0 : 1, film ?? 0, (film ?? 0) + c, 0, 0]));
      const Gm = W(name + '.weight'), Bt = W(name + '.bias');
      plan.push([P.stats, (cond) => [X, AB, Gm, Bt, cond, u], [groups, 1, 1]]);
    };
    const copy = (X, Y, n, off = 0) => {
      const u = uni(new Uint32Array([n, off, 0, 0]));
      plan.push([P.copy, () => [X, Y, u], [wg(n), 1, 1]]);
    };
    const convt = (X, name, Y, c, hin) => {
      const u = uni(new Uint32Array([c, c, hin, hin])), Wt = W(name + '.weight'), Bv = W(name + '.bias');
      plan.push([P.convt, () => [X, Wt, Bv, Y, u], [wg(c * 4 * hin * hin), 1, 1]]);
    };
    const tmp = {};
    const scratch = (res) => tmp[res] || (tmp[res] = { t: buf(200 * res * res), ab1: buf(400), ab2: buf(400) });
    const resblock = (X, Y, name, cin, cout, res) => {
      const s = scratch(res);
      stats(X, s.ab1, name + '.n1', cin, res, Math.min(8, Math.floor(cin / 4)));
      conv(X, name + '.c1', s.t, cin, cout, res, 3, 1, 1, { act: s.ab1 });
      this.films.push([w.get(name + '.film.weight'), w.get(name + '.film.bias'), cout, filmLen]);
      stats(s.t, s.ab2, name + '.n2', cout, res, Math.min(8, Math.floor(cout / 4)), filmLen);
      filmLen += 2 * cout;
      if (cin !== cout) {
        conv(X, name + '.skip', Y, cin, cout, res, 1, 1, 0);
        conv(s.t, name + '.c2', Y, cout, cout, res, 3, 1, 1, { act: s.ab2, accum: true });
      } else {
        conv(s.t, name + '.c2', Y, cout, cout, res, 3, 1, 1, { act: s.ab2, res: X });
      }
    };

    const B = (this.bufs = {
      inp: buf(2 * NPIX), out: buf(NPIX),
      read: device.createBuffer({ size: NPIX * 4, usage: GPUBufferUsage.MAP_READ | CD }),
    });
    const [c0, c1, c2] = ch;
    const A64 = buf(c0 * 4096), B64 = buf(c0 * 4096), S0 = buf(c0 * 4096);
    const A32 = buf(c0 * 1024), B32 = buf(c1 * 1024), S1 = buf(c1 * 1024);
    const A16 = buf(c1 * 256), B16 = buf(c2 * 256), S2 = buf(c2 * 256), M16 = buf(c2 * 256);
    const C16 = buf(2 * c2 * 256), D16 = buf(c2 * 256), E16 = buf(c2 * 256);
    const C32 = buf((c2 + c1) * 1024), D32 = buf(c1 * 1024), E32 = buf(c1 * 1024);
    const C64 = buf((c1 + c0) * 4096), D64 = buf(c0 * 4096), E64 = buf(c0 * 4096), abOut = buf(64);

    conv(B.inp, 'inp', A64, 6, c0, 64, 3, 1, 1);
    resblock(A64, B64, 'down.0.0', c0, c0, 64);
    resblock(B64, S0, 'down.0.1', c0, c0, 64);
    conv(S0, 'pool.0', A32, c0, c0, 64, 3, 2, 1);
    resblock(A32, B32, 'down.1.0', c0, c1, 32);
    resblock(B32, S1, 'down.1.1', c1, c1, 32);
    conv(S1, 'pool.1', A16, c1, c1, 32, 3, 2, 1);
    resblock(A16, B16, 'down.2.0', c1, c2, 16);
    resblock(B16, S2, 'down.2.1', c2, c2, 16);
    resblock(S2, M16, 'mid', c2, c2, 16);
    copy(M16, C16, c2 * 256, 0);
    copy(S2, C16, c2 * 256, c2 * 256);
    resblock(C16, D16, 'up.0.0', 2 * c2, c2, 16);
    resblock(D16, E16, 'up.0.1', c2, c2, 16);
    convt(E16, 'ups.0', C32, c2, 16);
    copy(S1, C32, c1 * 1024, c2 * 1024);
    resblock(C32, D32, 'up.1.0', c2 + c1, c1, 32);
    resblock(D32, E32, 'up.1.1', c1, c1, 32);
    convt(E32, 'ups.1', C64, c1, 32);
    copy(S0, C64, c0 * 4096, c1 * 4096);
    resblock(C64, D64, 'up.2.0', c1 + c0, c0, 64);
    resblock(D64, E64, 'up.2.1', c0, c0, 64);
    stats(E64, abOut, 'out_norm', c0, 64, 6);
    conv(E64, 'out', B.out, c0, 3, 64, 3, 1, 1, { act: abOut });

    this.condData = new Float32Array(filmLen);
    const bg = (p, list) => device.createBindGroup({
      layout: p.getBindGroupLayout(0), entries: list.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    // One materialized pass list (and Euler step) per denoising step, each with its own cond buffer,
    // so all steps of a frame can be encoded into a single submission.
    this.steps = [];
    this.materialize = (j) => {
      if (this.steps[j]) return this.steps[j];
      const cond = buf(filmLen);
      const eu = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | CD });
      return (this.steps[j] = {
        cond, eu,
        ops: plan.map(([p, f, n]) => [p, bg(p, f(cond)), n]),
        euler: bg(P.euler, [B.inp, B.out, eu]),
      });
    };
    this.reset();
  }

  reset() {
    this.n = 0;
    this.pending = null;
    this.device.queue.writeBuffer(this.bufs.inp, 3 * 4096 * 4, new Float32Array(3 * 4096));
  }

  _cond(t, action, actMask) {
    const D = this.cdim, c = this.cpu, emb = new Float32Array(256);
    for (let i = 0; i < 128; i++) {
      const a = t * 1000 * Math.exp((-Math.log(10000) * i) / 128);
      emb[i] = Math.cos(a); emb[128 + i] = Math.sin(a);
    }
    const lin = (k, x, nOut, nIn) => matvec(c[k][0], c[k][1], x, nOut, nIn);
    const temb = lin('t_mlp.2', silu(lin('t_mlp.0', emb, D, 256)), D, D);
    const aemb = actMask ? lin('a_mlp.2', silu(lin('a_mlp.0', action, D, action.length)), D, D) : c.null;
    const h = silu(temb.map((v, i) => v + aemb[i]));
    for (const [Wf, bf, cout, off] of this.films) this.condData.set(matvec(Wf, bf, h, 2 * cout, D), off);
    return this.condData;
  }

  _run(pass, s) {
    for (const [p, g, n] of s.ops) {
      pass.setPipeline(p);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(n[0], n[1], n[2]);
    }
  }

  async _read(enc, src) {
    const B = this.bufs;
    enc.copyBufferToBuffer(src, 0, B.read, 0, 3 * 4096 * 4);
    this.device.queue.submit([enc.finish()]);
    await B.read.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(B.read.getMappedRange().slice(0));
    B.read.unmap();
    return v;
  }

  // One network call: returns the velocity (or just records a cache-fill input).
  async step(x, t, action, actMask, wantOutput = true) {
    if (!wantOutput) { this.pending = x; return null; }
    const q = this.device.queue, s = this.materialize(0);
    q.writeBuffer(this.bufs.inp, 0, x);
    q.writeBuffer(s.cond, 0, this._cond(t, action, actMask));
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    this._run(pass, s);
    pass.end();
    return this._read(enc, this.bufs.out);
  }

  // A whole frame: the Euler steps over ts from the given noise, on the GPU, with one
  // readback. Returns x0 clamped to [-1, 1].
  async denoise(noise, ts, action) {
    const q = this.device.queue, nfe = ts.length - 1, B = this.bufs;
    if (nfe > MAX_NFE) throw new Error('too many steps');
    q.writeBuffer(B.inp, 0, noise);
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    for (let j = 0; j < nfe; j++) {
      const s = this.materialize(j), u = new ArrayBuffer(16);
      new Float32Array(u, 0, 1)[0] = ts[j + 1] - ts[j];
      new Uint32Array(u, 4, 1)[0] = j === nfe - 1 ? 1 : 0;
      q.writeBuffer(s.cond, 0, this._cond(ts[j], action, 1));
      q.writeBuffer(s.eu, 0, u);
      this._run(pass, s);
      pass.setPipeline(this.P.euler);
      pass.setBindGroup(0, s.euler);
      pass.dispatchWorkgroups(12288 / 64);
    }
    pass.end();
    return this._read(enc, B.inp);
  }

  commit() {
    this.device.queue.writeBuffer(this.bufs.inp, 3 * 4096 * 4, this.pending);
    this.n++;
  }

  destroy() { this.device.destroy(); }
}
