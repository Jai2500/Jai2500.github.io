// Module worker: loads the runtime and weights, then imagines rollouts.
// Messages in:  {type: 'init', manifest, ortBase, backends: ['webgpu', 'wasm']}
//               {type: 'run', seed}          a fresh strip (scripted policy, closed loop)
//               {type: 'step', path}         continue the current rollout by one frame from
//                                            visitor input: [[x, y]] (a click) or a path (a drag)
// Messages out: {type: 'ready', backend, ms, params}
//               {type: 'frame', seed, i, mask: Uint8Array(3*64*64)}   (i = 0 is the simulator frame)
//               {type: 'done', seed, ms, agent}
//               {type: 'iframe', seed, n, mask, agent, drift, goal}  (one interactive frame, t = 8n)
//               {type: 'error', message}

import {
  makeRng, sampleStripStart, renderFrame, actionFromTargets, steerTargets, maskPose, maskCentroid, GOAL, PX,
} from './world.js';

const STEPS = 6, NPIX = 3 * 64 * 64;
let model = null, man = null, st = null; // st: the rollout that 'step' continues

const fetchBytes = async (url) => {
  const r = await fetch(url, { priority: 'low' });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
};

// man.arch: 'dit' (causal DiT with a KV cache) or 'unet' (conv U-Net, previous frame as context).
async function load(kind, base, ortBase, allowFallback) {
  const unet = man.arch === 'unet';
  if (kind === 'webgpu') {
    const [mod, buf] = await Promise.all([unet ? import('./gpu_unet.js') : import('./gpu.js'), fetchBytes(base + man.weights)]);
    return (unet ? mod.GpuUNet : mod.GpuDiT).create(mod.parseWeights(buf), { allowFallback });
  }
  const [mod, buf] = await Promise.all([import('./ort.js'), fetchBytes(base + man.onnx)]);
  return (unet ? mod.OrtUNet : mod.OrtDiT).create(new Uint8Array(buf), man.cfg, ortBase);
}

// Try the backends in order (WebGPU first), then warm up: compile every kernel once.
async function init({ manifest, ortBase, backends, allowFallback }) {
  const t0 = performance.now();
  man = await (await fetch(manifest)).json();
  const base = new URL('.', new URL(manifest, self.location.href)).href;
  let err = null;
  for (const kind of backends) {
    try {
      model = await load(kind, base, ortBase, allowFallback);
      model.kind = kind;
      break;
    } catch (e) { err = e; }
  }
  if (!model) throw err || new Error('no backend');
  model.reset();
  await model.step(new Float32Array(NPIX), 1, new Float32Array(16), 1, true);
  return { backend: model.kind, ms: performance.now() - t0, params: man.params };
}

const toU8 = (m) => {
  const u = new Uint8Array(NPIX);
  for (let i = 0; i < NPIX; i++) u[i] = Math.round(Math.min(1, Math.max(0, m[i])) * 255);
  return u;
};
const blockArea = (m) => { let s = 0; for (let i = 0; i < 4096; i++) s += m[i]; return s; };
const randn = () => { const z = new Float32Array(NPIX); for (let i = 0; i < NPIX; i++) z[i] = st.noise.normal(); return z; };
const post = (msg, m) => { const u = toU8(m); self.postMessage({ ...msg, seed: st.seed, mask: u }, [u.buffer]); };

// Re-encode a finished frame at sigma_ctx and push it into the cache (free for the U-Net).
async function encode(x, action, actMask) {
  const sig = man.sigma_ctx, z = randn(), xc = new Float32Array(NPIX);
  for (let i = 0; i < NPIX; i++) xc[i] = (1 - sig) * x[i] + sig * z[i];
  await model.step(xc, sig, action, actMask, false);
  model.commit();
}

// One new frame from the current cache and an action chunk: Euler steps of the flow
// (on the GPU in one submission when the runtime supports it).
async function imagine(action) {
  const ts = man.schedule, x = randn();
  if (model.denoise) return model.denoise(x, ts, action);
  for (let j = 0; j + 1 < ts.length; j++) {
    const v = await model.step(x, ts[j], action, 1, true);
    const dt = ts[j + 1] - ts[j];
    for (let k = 0; k < NPIX; k++) x[k] += dt * v[k];
  }
  for (let k = 0; k < NPIX; k++) x[k] = Math.min(1, Math.max(-1, x[k]));
  return x;
}

async function run(seed) {
  const t0 = performance.now(), rng = makeRng(seed);
  st = { seed, noise: makeRng(seed ^ 0x9e3779b9), n: 0 };
  // t = 0: a real simulator frame, drawn with the training renderer.
  let { agent, block, policy } = sampleStripStart(rng);
  let m = renderFrame(agent, block);
  post({ type: 'frame', i: 0 }, m);
  st.area0 = blockArea(m);
  model.reset();
  await encode(m.map((v) => 2 * v - 1), new Float32Array(16), 0);
  for (let i = 1; i <= STEPS; i++) {
    const action = actionFromTargets(policy.plan(agent, block), agent);
    const x = await imagine(action);
    m = x.map((v) => (v + 1) / 2);
    post({ type: 'frame', i }, m);
    // Closed loop: the policy acts on the state read back from the imagined frame.
    block = maskPose(m, 0) || block;
    agent = maskCentroid(m, 1) || agent;
    if (i === STEPS) self.postMessage({ type: 'done', seed, ms: performance.now() - t0, agent });
    await encode(x, action, 1); // the last one lets a visitor continue the rollout
  }
  Object.assign(st, { n: STEPS, agent, target: policy.target });
}

async function step(path) {
  if (!st || st.n < STEPS) throw new Error('no rollout to continue');
  const targets = steerTargets(path, st.agent, st.target);
  st.target = [targets[targets.length - 2], targets[targets.length - 1]];
  const action = actionFromTargets(targets, st.agent);
  const x = await imagine(action);
  const m = x.map((v) => (v + 1) / 2);
  st.n++;
  st.agent = maskCentroid(m, 1) || st.agent;
  const drift = Math.abs(blockArea(m) - st.area0) > 0.1 * st.area0;
  // In the goal: imagined T within 3 px and 15 degrees of the outline.
  const pose = maskPose(m, 0);
  const dth = pose && Math.abs(Math.atan2(Math.sin(pose[2] - GOAL[2]), Math.cos(pose[2] - GOAL[2])));
  const goal = !!pose && Math.hypot(pose[0] - GOAL[0], pose[1] - GOAL[1]) < 3 * PX && dth < Math.PI / 12;
  post({ type: 'iframe', n: st.n, agent: st.agent, drift, goal }, m);
  await encode(x, action, 1);
}

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      if (data.type === 'init') self.postMessage({ type: 'ready', ...(await init(data)) });
      else if (data.type === 'run') await run(data.seed);
      else if (data.type === 'step') await step(data.path);
    } catch (e) {
      self.postMessage({ type: 'error', message: String((e && e.message) || e) });
    }
  });
};
