// Fallback backend for browsers without WebGPU: the exported ONNX denoiser
// call on ONNX Runtime Web's single-threaded wasm build (GitHub Pages cannot
// send the COOP/COEP headers that multithreading needs). Same interface as
// GpuDiT: reset(), step(x, t, action, actMask, wantOutput), commit().

export class OrtDiT {
  static async create(onnxBytes, cfg, ortBase) {
    const ort = await import(/* webpackIgnore: true */ ortBase + 'ort.wasm.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = ortBase;
    const session = await ort.InferenceSession.create(onnxBytes, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    return new OrtDiT(ort, session, cfg);
  }

  constructor(ort, session, cfg) {
    this.ort = ort;
    this.session = session;
    this.L = cfg.depth;
    this.N = (cfg.res / cfg.patch) ** 2;
    this.H = cfg.heads;
    this.usePrev = !!cfg.prev_frame;
    this.slot = this.N * cfg.width; // N * H * hd per layer
    this.reset();
  }

  reset() {
    this.n = 0;
    this.slots = [null, null, null]; // each {k, v}: Float32Array(L * slot)
    this.pending = null;
    this.prev = new Float32Array(3 * 4096); // input of the last committed call
  }

  _past(which) {
    const { L, slot } = this, out = new Float32Array(L * 3 * slot);
    this.slots.forEach((s, j) => {
      if (!s) return;
      for (let l = 0; l < L; l++) out.set(s[which].subarray(l * slot, (l + 1) * slot), (l * 3 + j) * slot);
    });
    return out;
  }

  async step(x, t, action, actMask, wantOutput = true) {
    const { ort, n, L, N, H, slot } = this, T = (d, s) => new ort.Tensor('float32', d, s);
    const shape = [L, 1, 3, N, H, slot / (N * H)];
    const feeds = {
      x: T(x, [1, 3, 64, 64]), t: T(new Float32Array([t]), [1]), action: T(action, [1, 16]),
      act_mask: T(new Float32Array([actMask]), [1]),
      past_k: T(this._past('k'), shape), past_v: T(this._past('v'), shape),
      ctx_valid: T(new Float32Array([n >= 1, n >= 3, n >= 2]), [1, 3]),
      ctx_pos: T(new Float32Array([Math.max(0, n - 7), n - 2, n - 1]), [1, 3]),
      cur_pos: T(new Float32Array([n]), [1]),
    };
    if (this.usePrev) feeds.prev = T(this.prev, [1, 3, 64, 64]);
    const out = await this.session.run(feeds);
    this.pending = { k: out.k.data, v: out.v.data, x };
    return wantOutput ? out.velocity.data : null;
  }

  commit() {
    this.prev = this.pending.x;
    if (this.n === 0) this.slots[0] = this.pending;
    else { this.slots[1] = this.slots[2]; this.slots[2] = this.pending; }
    this.n++;
  }

  destroy() { this.session.release?.(); }
}

// U-Net variant: the context is the input of the last committed call.
export class OrtUNet {
  static async create(onnxBytes, cfg, ortBase) {
    const ort = await import(/* webpackIgnore: true */ ortBase + 'ort.wasm.min.mjs');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = ortBase;
    const session = await ort.InferenceSession.create(onnxBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return new OrtUNet(ort, session);
  }

  constructor(ort, session) { this.ort = ort; this.session = session; this.reset(); }

  reset() { this.n = 0; this.ctx = new Float32Array(3 * 4096); this.pending = null; }

  async step(x, t, action, actMask, wantOutput = true) {
    if (!wantOutput) { this.pending = x; return null; }
    const T = (d, s) => new this.ort.Tensor('float32', d, s);
    const out = await this.session.run({
      x: T(x, [1, 3, 64, 64]), ctx: T(this.ctx, [1, 3, 64, 64]), t: T(new Float32Array([t]), [1]),
      action: T(action, [1, 16]), act_mask: T(new Float32Array([actMask]), [1]),
    });
    return out.velocity.data;
  }

  commit() { this.ctx = this.pending; this.n++; }

  destroy() { this.session.release?.(); }
}
