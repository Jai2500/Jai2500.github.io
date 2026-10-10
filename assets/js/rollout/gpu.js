// Hand-written WebGPU runtime for the tiny causal DiT (one denoiser call
// against a KV cache of [anchor, k-2, k-1]). Mirrors DiT.forward_step in
// tinywm/model.py. The per-frame conditioning MLP runs on the CPU; tokens,
// attention and the KV cache stay on the GPU.

import { parseWeights } from './weights.js';

export { parseWeights };

// ---------------------------------------------------------------- CPU conditioning
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

class Cond {
  constructor(w, cfg) {
    const D = cfg.width;
    this.D = D; this.L = cfg.depth;
    for (const k of ['t_mlp.0', 't_mlp.2', 'a_mlp.0', 'a_mlp.2', 'ada', 'final_ada']) {
      this[k] = [w.get(k + '.weight'), w.get(k + '.bias')];
    }
    this.nullAction = w.get('null_action');
    this.tables = Array.from({ length: this.L }, (_, l) => w.get(`blocks.${l}.table`));
    this.out = new Float32Array(this.L * 6 * D + 2 * D);
  }

  compute(t, action, actMask) {
    const D = this.D, emb = new Float32Array(256);
    for (let i = 0; i < 128; i++) {
      const a = t * 1000 * Math.exp((-Math.log(10000) * i) / 128);
      emb[i] = Math.cos(a); emb[128 + i] = Math.sin(a);
    }
    const lin = (k, x, nOut, nIn) => matvec(this[k][0], this[k][1], x, nOut, nIn);
    const temb = lin('t_mlp.2', silu(lin('t_mlp.0', emb, D, 256)), D, D);
    const aemb = actMask ? lin('a_mlp.2', silu(lin('a_mlp.0', action, D, action.length)), D, D) : this.nullAction;
    const c = silu(temb.map((v, i) => v + aemb[i]));
    const mod = lin('ada', c, 6 * D, D), fin = lin('final_ada', c, 2 * D, D);
    for (let l = 0; l < this.L; l++) {
      const tb = this.tables[l], o = l * 6 * D;
      for (let k = 0; k < 6 * D; k++) this.out[o + k] = mod[k] + tb[k];
    }
    this.out.set(fin, this.L * 6 * D);
    return this.out;
  }
}

// ---------------------------------------------------------------- WGSL
const common = (D, H, HD, N, KIN) => /* wgsl */ `
const D: u32 = ${D}u; const NH: u32 = ${H}u; const HD: u32 = ${HD}u; const N: u32 = ${N}u; const KIN: u32 = ${KIN}u;
const LSTRIDE: u32 = ${2 * H * N * HD}u;
struct Frame { cur: f32, p0: f32, p1: f32, p2: f32, v0: f32, v1: f32, v2: f32, t: f32 };
fn angle(i: u32, pos: f32, n: u32) -> f32 {
  if (i < 4u) { return pos * pow(100.0, -f32(i) / 4.0); }
  if (i < 10u) { return f32(n / 16u) * pow(100.0, -f32(i - 4u) / 6.0); }
  return f32(n % 16u) * pow(100.0, -f32(i - 10u) / 6.0);
}
`;

const PATCH = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> Bv: array<f32>;
@group(0) @binding(3) var<storage, read_write> Hs: array<f32>;
@group(0) @binding(4) var<storage, read> Pos: array<f32>;
@group(0) @binding(5) var<storage, read> Prev: array<f32>;
@compute @workgroup_size(D)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let n = wg.x; let d = l.x; let r = n / 16u; let c = n % 16u;
  var acc = Bv[d] + Pos[n * D + d];
  for (var j = 0u; j < 48u; j++) {
    let ch = j / 16u; let py = (j % 16u) / 4u; let px = j % 4u;
    let pix = ch * 4096u + (r * 4u + py) * 64u + c * 4u + px;
    acc += Wt[d * KIN + j] * X[pix];
    if (KIN > 48u) { acc += Wt[d * KIN + 48u + j] * Prev[pix]; }
  }
  Hs[n * D + d] = acc;
}`;

const NORM = `
struct P { shift: u32, scale: u32, a: u32, b: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> C: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, D>;
@compute @workgroup_size(D)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let i = wg.x * D + l.x; let v = X[i];
  red[l.x] = v * v;
  workgroupBarrier();
  for (var s = D / 2u; s > 0u; s >>= 1u) {
    if (l.x < s) { red[l.x] += red[l.x + s]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(D) + 1e-6);
  Y[i] = v * r * (1.0 + C[p.scale + l.x]) + C[p.shift + l.x];
}`;

// Y[M,N] = X[M,K] W[N,K]^T + b, modes: 0 store, 1 SiLU, 2 Y += gate*acc, 3 unpatchify.
const LINEAR = `
struct P { M: u32, K: u32, N: u32, mode: u32, gate: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> Bv: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> C: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> xs: array<f32, 256>;
var<workgroup> ws: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let row = wg.y * 16u + l.y; let col = wg.x * 16u + l.x;
  var acc = 0.0;
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    xs[l.y * 16u + l.x] = X[row * p.K + k0 + l.x];
    ws[l.y * 16u + l.x] = Wt[(wg.x * 16u + l.y) * p.K + k0 + l.x];
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc += xs[l.y * 16u + k] * ws[l.x * 16u + k]; }
    workgroupBarrier();
  }
  acc += Bv[col];
  if (p.mode == 0u) { Y[row * p.N + col] = acc; }
  else if (p.mode == 1u) { Y[row * p.N + col] = acc / (1.0 + exp(-acc)); }
  else if (p.mode == 2u) { Y[row * p.N + col] += C[p.gate + col] * acc; }
  else {
    let ch = col / 16u; let py = (col % 16u) / 4u; let px = col % 4u;
    let r = row / 16u; let c = row % 16u;
    Y[ch * 4096u + (r * 4u + py) * 64u + c * 4u + px] = acc;
  }
}`;

// x0-prediction models: velocity = (x_t - x0) / max(t, 0.05), as in DiT._final.
const XPRED = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> fr: Frame;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  Out[g.x] = (X[g.x] - Out[g.x]) / max(fr.t, 0.05);
}`;

const QKV_POST = `
struct P { layer: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read> QN: array<f32>;
@group(0) @binding(2) var<storage, read> KN: array<f32>;
@group(0) @binding(3) var<storage, read_write> Qo: array<f32>;
@group(0) @binding(4) var<storage, read_write> Kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> Pend: array<f32>;
@group(0) @binding(6) var<uniform> fr: Frame;
@group(0) @binding(7) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let n = g.x / NH; let h = g.x % NH;
  let bq = n * 3u * D + h * HD; let bk = bq + D; let bv = bq + 2u * D;
  var sq = 0.0; var sk = 0.0;
  for (var e = 0u; e < HD; e++) { sq += QKV[bq + e] * QKV[bq + e]; sk += QKV[bk + e] * QKV[bk + e]; }
  let rq = inverseSqrt(sq / f32(HD) + 1e-6); let rk = inverseSqrt(sk / f32(HD) + 1e-6);
  let o = h * N * HD + n * HD;
  let pk = p.layer * LSTRIDE + o; let pv = pk + NH * N * HD;
  let half = HD / 2u;
  for (var i = 0u; i < half; i++) {
    let a = angle(i, fr.cur, n); let c = cos(a); let s = sin(a);
    let q1 = QKV[bq + i] * rq * QN[i]; let q2 = QKV[bq + i + half] * rq * QN[i + half];
    Qo[o + i] = q1 * c - q2 * s; Qo[o + i + half] = q2 * c + q1 * s;
    let k1 = QKV[bk + i] * rk * KN[i]; let k2 = QKV[bk + i + half] * rk * KN[i + half];
    Pend[pk + i] = k1; Pend[pk + i + half] = k2;
    Kc[o + i] = k1 * c - k2 * s; Kc[o + i + half] = k2 * c + k1 * s;
  }
  for (var e = 0u; e < HD; e++) { Pend[pv + e] = QKV[bv + e]; }
}`;

const ATTN = `
struct P { layer: u32, a: u32, b: u32, c: u32 };
const TILE: u32 = 32u;
const TILE_HD: u32 = TILE * HD;
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> Kc: array<f32>;
@group(0) @binding(2) var<storage, read> Pend: array<f32>;
@group(0) @binding(3) var<storage, read> S0: array<f32>;
@group(0) @binding(4) var<storage, read> S1: array<f32>;
@group(0) @binding(5) var<storage, read> S2: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(7) var<uniform> fr: Frame;
@group(0) @binding(8) var<uniform> p: P;
var<workgroup> ks: array<f32, TILE_HD>;
var<workgroup> vs: array<f32, TILE_HD>;
fn slot(src: u32, i: u32) -> f32 {
  if (src == 0u) { return S0[i]; }
  if (src == 1u) { return S1[i]; }
  return S2[i];
}
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let h = wg.y; let n = wg.x * TILE + l.x;
  let scale = inverseSqrt(f32(HD));
  let half = HD / 2u;
  var q: array<f32, HD>;
  var acc: array<f32, HD>;
  var sc: array<f32, 32>;
  for (var e = 0u; e < HD; e++) { q[e] = Q[h * N * HD + n * HD + e] * scale; acc[e] = 0.0; }
  var m = -1e30; var lsum = 0.0;
  let valid = array<f32, 3>(fr.v0, fr.v1, fr.v2);
  let pos = array<f32, 3>(fr.p0, fr.p1, fr.p2);
  for (var src = 0u; src < 4u; src++) {
    if (src < 3u) { if (valid[src] == 0.0) { continue; } }
    for (var j0 = 0u; j0 < N; j0 += TILE) {
      workgroupBarrier();
      let j = j0 + l.x;
      let kb = p.layer * LSTRIDE + h * N * HD + j * HD;
      let vb = kb + NH * N * HD;
      if (src == 3u) {
        for (var e = 0u; e < HD; e++) {
          ks[l.x * HD + e] = Kc[h * N * HD + j * HD + e];
          vs[l.x * HD + e] = Pend[vb + e];
        }
      } else {
        for (var i = 0u; i < half; i++) {
          let a = angle(i, pos[src], j); let c = cos(a); let s = sin(a);
          let k1 = slot(src, kb + i); let k2 = slot(src, kb + i + half);
          ks[l.x * HD + i] = k1 * c - k2 * s; ks[l.x * HD + i + half] = k2 * c + k1 * s;
        }
        for (var e = 0u; e < HD; e++) { vs[l.x * HD + e] = slot(src, vb + e); }
      }
      workgroupBarrier();
      var tmax = -1e30;
      for (var jj = 0u; jj < TILE; jj++) {
        var s = 0.0;
        for (var e = 0u; e < HD; e++) { s += q[e] * ks[jj * HD + e]; }
        sc[jj] = s; tmax = max(tmax, s);
      }
      let mn = max(m, tmax); let corr = exp(m - mn);
      lsum *= corr;
      for (var e = 0u; e < HD; e++) { acc[e] *= corr; }
      for (var jj = 0u; jj < TILE; jj++) {
        let pj = exp(sc[jj] - mn); lsum += pj;
        for (var e = 0u; e < HD; e++) { acc[e] += pj * vs[jj * HD + e]; }
      }
      m = mn;
    }
  }
  for (var e = 0u; e < HD; e++) { O[n * D + h * HD + e] = acc[e] / lsum; }
}`;

// ---------------------------------------------------------------- runtime
export class GpuDiT {
  static async create(weights, opts = {}) {
    if (!navigator.gpu) throw new Error('no WebGPU');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no adapter');
    if (adapter.info?.isFallbackAdapter && !opts.allowFallback) throw new Error('fallback adapter');
    const device = await adapter.requestDevice();
    return new GpuDiT(device, weights);
  }

  constructor(device, w) {
    const cfg = w.cfg, D = cfg.width, H = cfg.heads, HD = D / H, L = cfg.depth, N = (cfg.res / cfg.patch) ** 2;
    if (cfg.patch !== 4 || cfg.res !== 64 || cfg.chans !== 3 || HD !== 32) throw new Error('unsupported config');
    Object.assign(this, { device, D, H, HD, L, N, mlp: cfg.mlp_ratio * D });
    this.cond = new Cond(w, cfg);
    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    const buf = (n, usage = S | CD | CS) => device.createBuffer({ size: Math.max(16, n * 4), usage });
    const upload = (arr) => {
      const b = device.createBuffer({ size: arr.byteLength, usage: S, mappedAtCreation: true });
      new Float32Array(b.getMappedRange()).set(arr);
      b.unmap();
      return b;
    };
    const uni = (arr, usage = GPUBufferUsage.UNIFORM | CD) => {
      const b = device.createBuffer({ size: arr.byteLength, usage, mappedAtCreation: true });
      new Uint32Array(b.getMappedRange()).set(arr);
      b.unmap();
      return b;
    };
    const kv = 2 * H * N * HD * L;
    const B = (this.bufs = {
      x: buf(3 * 4096), prev: buf(3 * 4096), h: buf(N * D), hn: buf(N * D), qkv: buf(N * 3 * D), q: buf(H * N * HD),
      kc: buf(H * N * HD), o: buf(N * D), m: buf(N * this.mlp), out: buf(3 * 4096),
      cond: buf(this.cond.out.length), pend: buf(kv), s0: buf(kv), s1: buf(kv), s2: buf(kv),
      frame: device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | CD }),
      read: device.createBuffer({ size: 3 * 4096 * 4, usage: GPUBufferUsage.MAP_READ | CD }),
    });
    const W = (name) => upload(w.get(name));
    const head = common(D, H, HD, N, cfg.prev_frame ? 96 : 48);

    const pipe = (src) => device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code: head + src }), entryPoint: 'main' },
    });
    const P = { patch: pipe(PATCH), norm: pipe(NORM), linear: pipe(LINEAR), post: pipe(QKV_POST), attn: pipe(ATTN) };
    if (cfg.pred === 'x0') P.xpred = pipe(XPRED);
    const bg = (p, list) => device.createBindGroup({
      layout: p.getBindGroupLayout(0), entries: list.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    // Dispatch list: [pipeline, bindGroup, x, y]
    const pos = cfg.pos_emb ? W('pos') : upload(new Float32Array(N * D));
    const pre = [[P.patch, bg(P.patch, [B.x, W('patch_embed.weight'), W('patch_embed.bias'), B.h, pos, B.prev]), N, 1]];
    const lin = (X, Wn, Y, M, K, Nn, mode, gate = 0) => [
      P.linear, bg(P.linear, [X, W(Wn + '.weight'), W(Wn + '.bias'), Y, B.cond, uni(new Uint32Array([M, K, Nn, mode, gate, 0, 0, 0]))]),
      Nn / 16, M / 16,
    ];
    const norm = (X, Y, shift, scale) => [P.norm, bg(P.norm, [X, Y, B.cond, uni(new Uint32Array([shift, scale, 0, 0]))]), N, 1];
    const body = [];
    for (let l = 0; l < L; l++) {
      const c = l * 6 * D, pl = `blocks.${l}.`, lu = uni(new Uint32Array([l, 0, 0, 0]));
      body.push(
        norm(B.h, B.hn, c, c + D),
        lin(B.hn, pl + 'qkv', B.qkv, N, D, 3 * D, 0),
        [P.post, bg(P.post, [B.qkv, W(pl + 'q_norm.weight'), W(pl + 'k_norm.weight'), B.q, B.kc, B.pend, B.frame, lu]), (N * H) / 64, 1],
        [P.attn, bg(P.attn, [B.q, B.kc, B.pend, B.s0, B.s1, B.s2, B.o, B.frame, lu]), N / 32, H],
        lin(B.o, pl + 'proj', B.h, N, D, D, 2, c + 2 * D),
        norm(B.h, B.hn, c + 3 * D, c + 4 * D),
        lin(B.hn, pl + 'fc1', B.m, N, D, this.mlp, 1),
        lin(B.m, pl + 'fc2', B.h, N, this.mlp, D, 2, c + 5 * D),
      );
    }
    const f = L * 6 * D;
    const post = [norm(B.h, B.hn, f, f + D), lin(B.hn, 'out', B.out, N, D, 48, 3)];
    if (P.xpred) post.push([P.xpred, bg(P.xpred, [B.x, B.out, B.frame]), (3 * 4096) / 64, 1]);
    this.passes = { pre, body, post };
    this.kvBytes = kv * 4;
    this.reset();
  }

  reset() {
    this.n = 0;
    this.device.queue.writeBuffer(this.bufs.prev, 0, new Float32Array(3 * 4096));
  }

  // One denoiser call against the current cache. Returns the velocity
  // (Float32Array 3*64*64) when wantOutput, else null (cache-fill pass).
  async step(x, t, action, actMask, wantOutput = true) {
    const { device, bufs: B, n } = this, q = device.queue;
    q.writeBuffer(B.x, 0, x);
    q.writeBuffer(B.cond, 0, this.cond.compute(t, action, actMask));
    q.writeBuffer(B.frame, 0, new Float32Array([n, Math.max(0, n - 7), n - 2, n - 1, n >= 1, n >= 3, n >= 2, t]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = (list) => list.forEach(([p, g, x, y]) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(x, y); });
    run(this.passes.pre);
    run(this.passes.body);
    if (wantOutput) run(this.passes.post);
    pass.end();
    if (wantOutput) enc.copyBufferToBuffer(B.out, 0, B.read, 0, 3 * 4096 * 4);
    q.submit([enc.finish()]);
    if (!wantOutput) return null;
    await B.read.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(B.read.getMappedRange().slice(0));
    B.read.unmap();
    return v;
  }

  // Push the K/V of the last call into the cache: anchor first, then a 2-frame window.
  // Its input becomes the previous-frame input of the following calls.
  commit() {
    const { device, bufs: B } = this, enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(B.x, 0, B.prev, 0, 3 * 4096 * 4);
    if (this.n === 0) enc.copyBufferToBuffer(B.pend, 0, B.s0, 0, this.kvBytes);
    else {
      enc.copyBufferToBuffer(B.s2, 0, B.s1, 0, this.kvBytes);
      enc.copyBufferToBuffer(B.pend, 0, B.s2, 0, this.kvBytes);
    }
    device.queue.submit([enc.finish()]);
    this.n++;
  }

  destroy() { this.device.destroy(); }
}
