// Toy Push-T world for the rollout strip: constants, supersampled mask
// renderer, start-state sampler, scripted chunk policy and mask-moment pose
// estimator. Mirrors tinywm/{world,render,sim,policy,metrics}.py.

export const WORLD = 512, RES = 64, SS = 4, PX = WORLD / RES;
export const S = 40, AGENT_R = 24, T_CY = (9.5 * S) / 7;
export const STRIDE = 8, ACT_SCALE = 128, MAX_SPEED = 30;
export const GOAL = [256, 256, Math.PI / 4], GOAL_STROKE = 8;
export const T_BOXES = [
  [-2 * S, 2 * S, -T_CY, S - T_CY],
  [-S / 2, S / 2, S - T_CY, 4 * S - T_CY],
];
export const T_POLY = [
  [-2 * S, -T_CY], [2 * S, -T_CY], [2 * S, S - T_CY], [S / 2, S - T_CY],
  [S / 2, 4 * S - T_CY], [-S / 2, 4 * S - T_CY], [-S / 2, S - T_CY], [-2 * S, S - T_CY],
];
export const T_RADIUS = Math.hypot(S / 2, 4 * S - T_CY);
const NS = RES * SS, PIX = RES * RES;

const wrap = (a) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
const rot = (th, x, y) => [Math.cos(th) * x - Math.sin(th) * y, Math.sin(th) * x + Math.cos(th) * y];
const clip = (x, y) => {
  const lo = AGENT_R, hi = WORLD - AGENT_R;
  return [Math.min(hi, Math.max(lo, x)), Math.min(hi, Math.max(lo, y))];
};

// Seeded RNG: mulberry32 uniforms, Box-Muller normals.
export function makeRng(seed) {
  let a = seed >>> 0, spare = null;
  const random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    const u = 1 - random(), v = random();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return { random, normal, uniform: (lo, hi) => lo + (hi - lo) * random() };
}

// ---------------------------------------------------------------- renderer
let goalCache = null;
function goalMask() {
  if (goalCache) return goalCache;
  const g = new Float32Array(PIX), [gx, gy, gth] = GOAL;
  const c = Math.cos(gth), s = Math.sin(gth), h = GOAL_STROKE / 2, w = 1 / (SS * SS);
  for (let i = 0; i < NS; i++) {
    const y = (i + 0.5) * (WORLD / NS) - gy;
    for (let j = 0; j < NS; j++) {
      const x = (j + 0.5) * (WORLD / NS) - gx;
      const qx = c * x + s * y, qy = -s * x + c * y;
      let d = 1e9;
      for (let k = 0; k < 8; k++) {
        const [ax, ay] = T_POLY[k], [bx, by] = T_POLY[(k + 1) % 8];
        const ex = bx - ax, ey = by - ay;
        const t = Math.min(1, Math.max(0, ((qx - ax) * ex + (qy - ay) * ey) / (ex * ex + ey * ey)));
        d = Math.min(d, Math.hypot(qx - ax - t * ex, qy - ay - t * ey));
      }
      if (d < h) g[((i / SS) | 0) * RES + ((j / SS) | 0)] += w;
    }
  }
  return (goalCache = g);
}

// Masks (block, agent, goal) as Float32Array(3*64*64) in [0, 1].
export function renderFrame(agent, block) {
  const out = new Float32Array(3 * PIX);
  const c = Math.cos(block[2]), s = Math.sin(block[2]), w = 1 / (SS * SS), r2 = AGENT_R * AGENT_R;
  for (let i = 0; i < NS; i++) {
    const y = (i + 0.5) * (WORLD / NS), dy = y - block[1], ay = y - agent[1];
    const row = ((i / SS) | 0) * RES;
    for (let j = 0; j < NS; j++) {
      const x = (j + 0.5) * (WORLD / NS), dx = x - block[0];
      const qx = c * dx + s * dy, qy = -s * dx + c * dy;
      const p = row + ((j / SS) | 0);
      for (const [x0, x1, y0, y1] of T_BOXES) {
        if (qx > x0 && qx < x1 && qy > y0 && qy < y1) { out[p] += w; break; }
      }
      const ax = x - agent[0];
      if (ax * ax + ay * ay < r2) out[PIX + p] += w;
    }
  }
  out.set(goalMask(), 2 * PIX);
  return out;
}

// ---------------------------------------------------------------- start state
function distToT(p, pose) {
  const c = Math.cos(pose[2]), s = Math.sin(pose[2]);
  const dx = p[0] - pose[0], dy = p[1] - pose[1];
  const qx = c * dx + s * dy, qy = -s * dx + c * dy;
  const inside = T_BOXES.some(([x0, x1, y0, y1]) => qx > x0 && qx < x1 && qy > y0 && qy < y1);
  let d = 1e9;
  for (let k = 0; k < 8; k++) {
    const [ax, ay] = T_POLY[k], [bx, by] = T_POLY[(k + 1) % 8];
    const ex = bx - ax, ey = by - ay;
    const t = Math.min(1, Math.max(0, ((qx - ax) * ex + (qy - ay) * ey) / (ex * ex + ey * ey)));
    d = Math.min(d, Math.hypot(qx - ax - t * ex, qy - ay - t * ey));
  }
  return inside ? -d : d;
}

export function sampleInitialState(rng, near = null) {
  const m = T_RADIUS + 12;
  const block = [rng.uniform(m, WORLD - m), rng.uniform(m, WORLD - m), rng.uniform(-Math.PI, Math.PI)];
  const lo = AGENT_R + 8, hi = WORLD - AGENT_R - 8;
  if (near === null) near = rng.random() < 0.5;
  for (;;) {
    const agent = [rng.uniform(lo, hi), rng.uniform(lo, hi)];
    const d = distToT(agent, block);
    if (d > AGENT_R + 8 && (!near || d < AGENT_R + 80)) return { agent, block };
  }
}

// ---------------------------------------------------------------- policy
const EDGES = T_POLY.map(([ax, ay], i) => {
  const [bx, by] = T_POLY[(i + 1) % 8], len = Math.hypot(bx - ax, by - ay);
  return [ax, ay, bx, by, len, (by - ay) / len, -(bx - ax) / len];
});
const PERIM = EDGES.reduce((a, e) => a + e[4], 0);

function sampleContact(rng) {
  let u = rng.uniform(0, PERIM);
  for (const [ax, ay, bx, by, ln, nx, ny] of EDGES) {
    if (u <= ln) { const t = u / ln; return [ax + t * (bx - ax), ay + t * (by - ay), nx, ny]; }
    u -= ln;
  }
  const e = EDGES[7];
  return [e[2], e[3], e[5], e[6]];
}

export class ChunkPolicy {
  constructor(rng, mode) {
    this.rng = rng; this.mode = mode; this.prim = null; this.target = null; this.ou = [0, 0];
  }

  _noise(sigma) {
    this.ou[0] = 0.8 * this.ou[0] + sigma * this.rng.normal();
    this.ou[1] = 0.8 * this.ou[1] + sigma * this.rng.normal();
    return this.ou;
  }

  _contactOk(b, [cx, cy, nx, ny], d) {
    const [wx, wy] = rot(b[2], cx + nx * (AGENT_R + 6), cy + ny * (AGENT_R + 6));
    const px = b[0] + wx, py = b[1] + wy, lo = AGENT_R, hi = WORLD - AGENT_R;
    if (!(lo <= px && px <= hi && lo <= py && py <= hi)) return false;
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const [vx, vy] of T_POLY) {
      const [ox, oy] = rot(b[2], vx, vy);
      x0 = Math.min(x0, b[0] + ox); x1 = Math.max(x1, b[0] + ox);
      y0 = Math.min(y0, b[1] + oy); y1 = Math.max(y1, b[1] + oy);
    }
    const [dx, dy] = rot(b[2], d[0], d[1]);
    if ((dx < -0.3 && x0 < 24) || (dx > 0.3 && x1 > WORLD - 24)) return false;
    if ((dy < -0.3 && y0 < 24) || (dy > 0.3 && y1 > WORLD - 24)) return false;
    return true;
  }

  _pushDir(c) {
    const r = this.rng, nx = c[2], ny = c[3];
    if (r.random() < 0.6) return [-nx, -ny];
    const sgn = r.random() < 0.5 ? 1 : -1;
    const dx = sgn * ny - nx, dy = -sgn * nx - ny, n = Math.hypot(dx, dy);
    return [dx / n, dy / n];
  }

  _newContactPrim(b, goalDirected) {
    const r = this.rng;
    let c, d;
    if (goalDirected) {
      let best = null, bestS = -1e9;
      const gx = GOAL[0] - b[0], gy = GOAL[1] - b[1], gd = Math.hypot(gx, gy) + 1e-6;
      const dth = wrap(GOAL[2] - b[2]);
      for (let i = 0; i < 16; i++) {
        const cc = sampleContact(r);
        const [wnx, wny] = rot(b[2], -cc[2], -cc[3]);
        const [rx, ry] = rot(b[2], cc[0], cc[1]);
        const trans = ((wnx * gx + wny * gy) / gd) * Math.min(1, gd / 60);
        const torque = (rx * wny - ry * wnx) / T_RADIUS;
        let s = trans + torque * Math.max(-1, Math.min(1, dth / 0.5)) + 0.3 * r.normal();
        if (!this._contactOk(b, cc, [-cc[2], -cc[3]])) s -= 10;
        if (s > bestS) { best = cc; bestS = s; }
      }
      c = best; d = [-c[2], -c[3]];
    } else {
      const cands = [];
      let cc, dd;
      for (let i = 0; i < 8; i++) {
        cc = sampleContact(r); dd = this._pushDir(cc);
        if (this._contactOk(b, cc, dd)) { cands.push([cc, dd]); if (cands.length === 4) break; }
      }
      if (!cands.length) cands.push([cc, dd]);
      let pick = cands[0];
      if (r.random() < 0.6) {
        const [tx, ty] = this.target;
        const cost = ([q]) => {
          const [wx, wy] = rot(b[2], q[0] + q[2] * AGENT_R, q[1] + q[3] * AGENT_R);
          return Math.hypot(b[0] + wx - tx, b[1] + wy - ty);
        };
        pick = cands.reduce((m, x) => (cost(x) < cost(m) ? x : m));
      }
      [c, d] = pick;
    }
    return {
      kind: 'approach', c: [c[0], c[1]], n: [c[2], c[3]], d,
      vApp: r.uniform(16, MAX_SPEED), vPush: r.uniform(6, MAX_SPEED * 0.8),
      pushSteps: Math.floor(r.uniform(24, 57)), route: r.random() < 0.7,
    };
  }

  _newRandomPrim() {
    const r = this.rng, u = r.random();
    if (u < 0.45) {
      const lo = AGENT_R + 4, hi = WORLD - AGENT_R - 4;
      return { kind: 'goto', p: [r.uniform(lo, hi), r.uniform(lo, hi)], v: r.uniform(4, MAX_SPEED) };
    }
    if (u < 0.85) {
      return { kind: 'wander', a: r.uniform(-Math.PI, Math.PI), v: r.uniform(4, MAX_SPEED), steps: Math.floor(r.uniform(8, 25)) };
    }
    return { kind: 'idle', steps: Math.floor(r.uniform(4, 13)) };
  }

  _nextPrim(b) {
    if (this.mode === 'random') return this._newRandomPrim();
    const u = this.rng.random();
    if (u < 0.06) return { kind: 'idle', steps: Math.floor(this.rng.uniform(4, 13)) };
    if (u < 0.1) return this._newRandomPrim();
    return this._newContactPrim(b, this.mode === 'goal');
  }

  _step(b) {
    const p = this.prim, [tx, ty] = this.target;
    switch (p.kind) {
      case 'idle':
        if (--p.steps <= 0) this.prim = null;
        return [tx, ty];
      case 'goto': {
        const dx = p.p[0] - tx, dy = p.p[1] - ty, d = Math.hypot(dx, dy);
        if (d <= p.v) { this.prim = null; return p.p; }
        return [tx + (dx / d) * p.v, ty + (dy / d) * p.v];
      }
      case 'wander': {
        p.a += 0.25 * this.rng.normal();
        const nx = tx + Math.cos(p.a) * p.v, ny = ty + Math.sin(p.a) * p.v, lo = AGENT_R, hi = WORLD - AGENT_R;
        if (!(lo < nx && nx < hi)) p.a = Math.PI - p.a;
        if (!(lo < ny && ny < hi)) p.a = -p.a;
        if (--p.steps <= 0) this.prim = null;
        return clip(nx, ny);
      }
      case 'approach': {
        const [cx, cy] = rot(b[2], p.c[0], p.c[1]), [nx, ny] = rot(b[2], p.n[0], p.n[1]);
        const off = AGENT_R + 6, px = b[0] + cx + nx * off, py = b[1] + cy + ny * off;
        let gx = px, gy = py;
        if (p.route) {
          const ax = tx - b[0], ay = ty - b[1], bx = px - b[0], by = py - b[1];
          if (ax * bx + ay * by < 0 && Math.hypot(ax, ay) < T_RADIUS + 60) {
            let mx = ax + bx, my = ay + by;
            if (Math.hypot(mx, my) < 1e-3) { mx = -ay; my = ax; }
            const m = Math.hypot(mx, my), rr = T_RADIUS + AGENT_R + 10;
            gx = b[0] + (mx / m) * rr; gy = b[1] + (my / m) * rr;
          }
        }
        const dx = gx - tx, dy = gy - ty, d = Math.hypot(dx, dy), v = p.vApp;
        if (d <= v) {
          if (gx === px && gy === py) p.kind = 'push';
          return clip(gx, gy);
        }
        return clip(tx + (dx / d) * v, ty + (dy / d) * v);
      }
      case 'push': {
        const [dx, dy] = rot(b[2], p.d[0], p.d[1]);
        if (--p.pushSteps <= 0) this.prim = null;
        return clip(tx + dx * p.vPush, ty + dy * p.vPush);
      }
    }
    throw new Error(p.kind);
  }

  // STRIDE commanded targets from the state at the chunk start: Float64Array(2*STRIDE).
  plan(agent, block) {
    if (this.target === null) this.target = [agent[0], agent[1]];
    const out = new Float64Array(2 * STRIDE), p = this.prim;
    if (p && p.kind === 'push') {
      const [cx, cy] = rot(block[2], p.c[0], p.c[1]);
      if (Math.hypot(block[0] + cx - this.target[0], block[1] + cy - this.target[1]) > 80) p.kind = 'approach';
    }
    for (let i = 0; i < STRIDE; i++) {
      if (this.prim === null) this.prim = this._nextPrim(block);
      this.target = this._step(block);
      const nz = this._noise(2.5);
      const [x, y] = clip(this.target[0] + nz[0], this.target[1] + nz[1]);
      out[2 * i] = x; out[2 * i + 1] = y;
    }
    return out;
  }
}

// Start for the homepage strip: the T in the central band, away from the goal, and the
// agent at rest just behind a goal-directed contact point (a subset of the 'near' training
// starts). Returns {agent, block, policy} with the policy already set on that contact.
export function sampleStripStart(rng) {
  for (;;) {
    const block = [rng.uniform(170, 342), rng.uniform(170, 342), rng.uniform(-Math.PI, Math.PI)];
    if (Math.hypot(block[0] - GOAL[0], block[1] - GOAL[1]) < 90) continue;
    const policy = new ChunkPolicy(rng, 'goal');
    policy.target = [0, 0];
    const prim = policy._newContactPrim(block, true);
    const off = AGENT_R + 6 + rng.uniform(10, 40);
    const [wx, wy] = rot(block[2], prim.c[0] + prim.n[0] * off, prim.c[1] + prim.n[1] * off);
    const agent = [block[0] + wx, block[1] + wy], lo = AGENT_R + 8, hi = WORLD - AGENT_R - 8;
    if (lo <= agent[0] && agent[0] <= hi && lo <= agent[1] && agent[1] <= hi && distToT(agent, block) > AGENT_R + 8) {
      policy.target = agent;
      policy.prim = prim;
      return { agent, block, policy };
    }
  }
}

// Visitor input -> STRIDE commanded targets (Float64Array(2*STRIDE)), in distribution with the
// scripted data: one point = go straight toward it at a push-like speed (the 'goto' primitive);
// a path = follow it, each control step capped at MAX_SPEED. Starts from the last commanded
// target when the agent is still close to it, as a continuing chunk would.
export const STEER_SPEED = 20;
export function steerTargets(path, agent, prev) {
  const out = new Float64Array(2 * STRIDE);
  let [tx, ty] = prev && Math.hypot(prev[0] - agent[0], prev[1] - agent[1]) < 80 ? prev : agent;
  for (let i = 0; i < STRIDE; i++) {
    const [gx, gy] = path[Math.min(i, path.length - 1)];
    const v = path.length === 1 ? STEER_SPEED : MAX_SPEED;
    const dx = gx - tx, dy = gy - ty, d = Math.hypot(dx, dy);
    [tx, ty] = d <= v ? clip(gx, gy) : clip(tx + (dx / d) * v, ty + (dy / d) * v);
    out[2 * i] = tx; out[2 * i + 1] = ty;
  }
  return out;
}

// Action chunk for the model: targets relative to the agent, scaled.
export function actionFromTargets(targets, agent) {
  const a = new Float32Array(2 * STRIDE);
  for (let i = 0; i < STRIDE; i++) {
    a[2 * i] = (targets[2 * i] - agent[0]) / ACT_SCALE;
    a[2 * i + 1] = (targets[2 * i + 1] - agent[1]) / ACT_SCALE;
  }
  return a;
}

// ---------------------------------------------------------------- state from masks
// Centroid (world units) of channel ch, or null if (nearly) empty.
export function maskCentroid(m, ch) {
  let s = 0, sx = 0, sy = 0;
  for (let i = 0; i < RES; i++) for (let j = 0; j < RES; j++) {
    const w = m[ch * PIX + i * RES + j];
    s += w; sx += w * (j + 0.5); sy += w * (i + 0.5);
  }
  return s < 1 ? null : [(sx / s) * PX, (sy / s) * PX];
}

// Block pose (world units, radians) from first to third moments of the mask.
export function maskPose(m, ch = 0) {
  let s = 0, cx = 0, cy = 0;
  for (let i = 0; i < RES; i++) for (let j = 0; j < RES; j++) {
    const w = m[ch * PIX + i * RES + j];
    s += w; cx += w * (j + 0.5); cy += w * (i + 0.5);
  }
  if (s < 8) return null;
  cx /= s; cy /= s;
  let xx = 0, yy = 0, xy = 0;
  for (let i = 0; i < RES; i++) for (let j = 0; j < RES; j++) {
    const w = m[ch * PIX + i * RES + j], dx = j + 0.5 - cx, dy = i + 0.5 - cy;
    xx += w * dx * dx; yy += w * dy * dy; xy += w * dx * dy;
  }
  const a = 0.5 * Math.atan2(2 * xy, xx - yy);
  let ux = Math.cos(a), uy = Math.sin(a), m3 = 0;
  for (let i = 0; i < RES; i++) for (let j = 0; j < RES; j++) {
    const w = m[ch * PIX + i * RES + j], p = (j + 0.5 - cx) * ux + (i + 0.5 - cy) * uy;
    m3 += w * p * p * p;
  }
  if (m3 < 0) { ux = -ux; uy = -uy; }
  return [cx * PX, cy * PX, Math.atan2(-ux, uy)];
}
