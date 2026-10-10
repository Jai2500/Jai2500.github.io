// "Push it yourself": the newest frame becomes a play area. Each deliberate input
// commits one action chunk (0.8 s of world time) and streams one new frame; the
// strip scrolls left like a ring buffer, for as long as the visitor keeps going.
//   click / tap        push toward that point
//   press and drag     follow the pointer, one chunk per 0.8 s while held
//   arrows + Enter     move a target cursor, push toward it
//   Esc, click outside, focus leaving, hidden tab, 30 s idle: stop
// Loaded lazily by loader.js on first entry, so it costs nothing until used.

const WORLD = 512, BEAT = 800, IDLE = 30000, WATCHDOG = 8000;

// Shown after a session, one per session, cycling (the position survives reloads).
// {n}: pushes, {s}: seconds of imagined world time.
const NOTES = [
  // favourites
  'a world model is a very confident guesser',
  'no T-blocks were harmed',
  'close enough for a world model',
  'you are now a world-model tester',
  'that push was out of distribution. probably',
  // same dry, ML-flavoured register
  'every frame after t = 0 was a well-behaved hallucination',
  'thank you for exploring the action space',
  'physics, approximately',
  'the agent never touched the T. it only imagined it',
  'you pushed it {n}× and the simulator was not consulted once',
  'gravity: off. friction: imagined',
  'the goal outline remains unimpressed',
  'somewhere, a real robot is jealous',
  'computed in your browser; no servers were bothered',
  // lighter jokes, hints, facts
  'the T has seen things',
  'try pushing it into the goal outline',
  'the T would like a word',
  'Push-T, but make it imaginary',
  '{n} pushes, {s} seconds of imagined time',
  'each push is 0.8 s of world time, dreamt up on demand',
  'fun fact: the model only ever sees the previous frame',
  'your actions, its imagination',
  // plainest last
  'imagined, not simulated',
  'nice pushing',
];
const GOAL_NOTE = 'you pushed the T into the goal. the world model is very proud';

function nextNote(pushes, goal) {
  if (goal) return GOAL_NOTE;
  let i = 0;
  try {
    i = (+localStorage.getItem('rollout-note') || 0) % NOTES.length;
    localStorage.setItem('rollout-note', i + 1);
  } catch { i = Math.floor(Math.random() * NOTES.length); }
  return NOTES[i].replace('{n}', pushes).replace('{s}', (pushes * 0.8).toFixed(1));
}

// Resample a pointer polyline to k points evenly spaced by arc length (constant speed).
function resample(pts, k = 8) {
  const seg = [];
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    seg.push(d);
    total += d;
  }
  if (total < 4) return [pts[pts.length - 1]];
  const out = [];
  for (let j = 1, i = 0, acc = 0; j <= k; j++) {
    const s = (total * j) / k;
    while (i < seg.length - 1 && acc + seg[i] < s) acc += seg[i++];
    const f = seg[i] ? (s - acc) / seg[i] : 1;
    out.push([pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1])]);
  }
  return out;
}

// The loader owns the frame ring (it applies every 'iframe' message, even one that lands after
// exit); this module only turns input into 'step' messages and tracks what is in flight.
export function startPlay({ fig, worker, agent, setStatus, fail, onExit }) {
  const box = fig.querySelector('.rollout-frames');
  const slot = fig.querySelector('.rollout-cell:last-child .rollout-slot');
  const cursor = document.createElement('span');
  const label = box.getAttribute('aria-label');
  let busy = false, pending = null, pushes = 0, drag = null, beat = 0, idle = 0, watchdog = 0, goal = false;
  let target = agent ? [...agent] : [WORLD / 2, WORLD / 2];

  cursor.className = 'rollout-cursor';
  cursor.hidden = true;
  slot.append(cursor);
  fig.classList.add('rollout-play');
  box.setAttribute('role', 'group');
  box.setAttribute('aria-label', 'Interactive rollout: the newest frame is the play area.');
  slot.tabIndex = 0;
  slot.setAttribute('role', 'application');
  slot.setAttribute('aria-label', 'Play area. Click or tap to push the agent toward a point, or press and drag. '
    + 'Arrow keys move the target and Enter pushes. Escape stops.');
  slot.focus({ preventScroll: true });
  setStatus('your turn · click or drag in the highlighted frame (arrows + Enter work too) · Esc to stop · ');

  const bump = () => { clearTimeout(idle); idle = setTimeout(exit, IDLE); };
  const show = ([x, y]) => {
    cursor.hidden = false;
    cursor.style.left = `${(x / WORLD) * 100}%`;
    cursor.style.top = `${(y / WORLD) * 100}%`;
  };
  const toWorld = (e) => {
    const r = slot.getBoundingClientRect(), c = (v) => Math.min(1, Math.max(0, v));
    return [c((e.clientX - r.left) / r.width) * WORLD, c((e.clientY - r.top) / r.height) * WORLD];
  };
  // One input in flight; while busy, keep only the latest (inputs never pile up).
  const send = (path) => {
    bump();
    if (busy) { pending = path; return; }
    busy = true;
    fig.classList.add('rollout-busy');
    watchdog = setTimeout(fail, WATCHDOG);
    worker.postMessage({ type: 'step', path });
  };

  const onMessage = ({ data: m }) => {
    if (m.type !== 'iframe') return;
    clearTimeout(watchdog);
    pushes++;
    if (m.goal && !goal) setStatus(`${GOAL_NOTE} · Esc to stop · `);
    else if (m.drift) setStatus('the block is drifting from its true shape · Esc to stop, or start over · ');
    goal = goal || m.goal;
    busy = false;
    fig.classList.remove('rollout-busy');
    if (pending) { const p = pending; pending = null; send(p); }
  };

  const down = (e) => {
    if (e.button > 0) return;
    e.preventDefault();
    slot.setPointerCapture(e.pointerId);
    const p = toWorld(e);
    show(p);
    drag = { pts: [p], sent: false };
    clearInterval(beat);
    beat = setInterval(() => {
      if (!drag) return;
      send(resample(drag.pts));
      drag.sent = true;
      drag.pts = [drag.pts[drag.pts.length - 1]];
    }, BEAT);
    bump();
  };
  const move = (e) => {
    if (!drag) return;
    const p = toWorld(e);
    show(p);
    drag.pts.push(p);
  };
  const up = () => {
    clearInterval(beat);
    if (drag && !drag.sent) send([drag.pts[drag.pts.length - 1]]);
    drag = null;
  };
  const key = (e) => {
    const d = e.shiftKey ? 48 : 16;
    const moves = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] };
    if (moves[e.key]) {
      target = target.map((v, i) => Math.min(WORLD, Math.max(0, v + moves[e.key][i])));
      show(target);
      bump();
    } else if (e.key === 'Enter' || e.key === ' ') {
      send([target]);
      show(target);
    } else if (e.key === 'Escape') exit();
    else return;
    e.preventDefault();
  };
  const outside = (e) => { if (!fig.contains(e.target)) exit(); };
  const blur = (e) => { if (e.relatedTarget && !fig.contains(e.relatedTarget)) exit(); };
  const hidden = () => { if (document.hidden) exit(); };

  worker.addEventListener('message', onMessage);
  slot.addEventListener('pointerdown', down);
  slot.addEventListener('pointermove', move);
  slot.addEventListener('pointerup', up);
  slot.addEventListener('pointercancel', up);
  slot.addEventListener('keydown', key);
  fig.addEventListener('focusout', blur);
  document.addEventListener('pointerdown', outside, true);
  document.addEventListener('visibilitychange', hidden);
  bump();

  let done = false;
  function exit() {
    if (done) return;
    done = true;
    clearInterval(beat);
    clearTimeout(idle);
    clearTimeout(watchdog);
    worker.removeEventListener('message', onMessage);
    slot.removeEventListener('pointerdown', down);
    slot.removeEventListener('pointermove', move);
    slot.removeEventListener('pointerup', up);
    slot.removeEventListener('pointercancel', up);
    slot.removeEventListener('keydown', key);
    fig.removeEventListener('focusout', blur);
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('visibilitychange', hidden);
    cursor.remove();
    slot.removeAttribute('tabindex');
    slot.removeAttribute('role');
    slot.removeAttribute('aria-label');
    box.setAttribute('role', 'img');
    box.setAttribute('aria-label', label);
    fig.classList.remove('rollout-play', 'rollout-busy');
    onExit({ pushes, note: pushes ? nextNote(pushes, goal) : '' });
  }
  return { exit };
}
