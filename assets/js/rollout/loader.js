// Rollout strip loader: upgrades the static placeholder figure to a rollout
// imagined in the visitor's browser. Runs after load, when idle, once the strip
// is on screen, and only if the gate passes. Any failure keeps the placeholder.
const fig = document.querySelector('.rollout-strip[data-manifest]');
const TIMEOUT = +(fig && fig.dataset.timeout) || 10000;

const loaded = () => new Promise((r) => (document.readyState === 'complete' ? r() : addEventListener('load', r, { once: true })));
const idle = () => new Promise((r) => (self.requestIdleCallback ? requestIdleCallback(r, { timeout: 2000 }) : setTimeout(r, 300)));
const onScreen = (el) => new Promise((r) => {
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); r(); } });
  io.observe(el);
});
const shown = () => new Promise((r) => {
  if (!document.hidden) return r();
  const f = () => { if (!document.hidden) { removeEventListener('visibilitychange', f); r(); } };
  addEventListener('visibilitychange', f);
});

function backends() {
  const n = navigator, c = n.connection;
  if ((c && c.saveData) || (n.deviceMemory && n.deviceMemory < 2)) return [];
  const list = [];
  if ('gpu' in n) list.push('webgpu');
  const mobile = n.userAgentData ? n.userAgentData.mobile : /Mobi|Android/i.test(n.userAgent);
  if (typeof WebAssembly === 'object' && !mobile && (n.hardwareConcurrency || 0) >= 4) list.push('wasm');
  return list;
}

// Theme colors as RGB, translucent ones composited over the page background.
function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name, fb) => cs.getPropertyValue(name).trim() || fb;
  const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  ctx.canvas.width = ctx.canvas.height = 1;
  const rgb = (col, under) => {
    ctx.globalCompositeOperation = 'copy';
    ctx.fillStyle = under; ctx.fillRect(0, 0, 1, 1);
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = col; ctx.fillRect(0, 0, 1, 1);
    return ctx.getImageData(0, 0, 1, 1).data;
  };
  const bg = rgb(v('--global-bg-color', '#fff'), '#fff');
  const under = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
  return [bg, rgb(v('--global-divider-color', '#ddd'), under), rgb(v('--global-theme-color', '#b509ac'), under),
    rgb(v('--global-text-color', '#000'), under)];
}

function paint(canvas, mask, colors) {
  const ctx = canvas.getContext('2d'), img = ctx.createImageData(64, 64), d = img.data;
  const [bg, goal, block, agent] = colors, P = 4096;
  for (let p = 0; p < P; p++) {
    const mg = mask[2 * P + p] / 255, mb = mask[p] / 255, ma = mask[P + p] / 255;
    for (let c = 0; c < 3; c++) {
      let x = bg[c];
      x += (goal[c] - x) * mg;
      x += (block[c] - x) * mb;
      x += (agent[c] - x) * ma;
      d[4 * p + c] = x;
    }
    d[4 * p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function boot(list) {
  const canvases = [...fig.querySelectorAll('canvas')];
  const labels = [...fig.querySelectorAll('.rollout-t')];
  const status = fig.querySelector('.rollout-status');
  const again = fig.querySelector('.rollout-again');
  const playBtn = fig.querySelector('.rollout-play-btn');
  const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let colors = themeColors(), worker = null, timer = 0, seed = 0, shownSeed = 0, shown = [], next = [];
  let play = null, agent = null, params = '';
  // "A rollout imagined by a 1.5M-param world model for this visit #4821 · <note> · ↻ … · ◎ …"
  // (parameter count from the manifest; the seed as a small muted tag).
  const caption = (note = '', end = ' \u00b7 ') => {
    const tag = document.createElement('span');
    tag.className = 'rollout-seed';
    tag.textContent = `#${shownSeed}`;
    status.replaceChildren(`A rollout imagined by a ${params}world model for this visit `, tag,
      note ? ` \u00b7 ${note}` : '', end);
  };

  const setLabels = (n) => labels.forEach((l, i) => { l.textContent = `t = ${8 * (n - labels.length + 1 + i)}`; });
  const repaint = () => canvases.forEach((c, i) => {
    if (shown[i]) paint(c, shown[i], colors);
    else c.classList.remove('live');
  });
  // Errors and timeouts: stop, and fall back to the last complete strip (or the placeholder).
  const fail = () => {
    clearTimeout(timer);
    if (play) play.exit();
    if (worker) worker.terminate();
    worker = null;
    again.hidden = playBtn.hidden = true;
    fig.classList.remove('rollout-live');
    if (shown.length) caption('', '');
    else fig.classList.remove('rollout-has-live');
    repaint();
  };
  const run = () => {
    if (play) play.exit();
    seed = Math.floor(Math.random() * 10000);
    next = [];
    setLabels(labels.length - 1);
    clearTimeout(timer);
    timer = setTimeout(fail, TIMEOUT);
    again.disabled = true;
    playBtn.hidden = true;
    fig.classList.remove('rollout-live');
    worker.postMessage({ type: 'run', seed });
  };
  // "Push it yourself": the interaction code loads only when someone opts in.
  const enterPlay = async () => {
    if (play || !worker || !fig.classList.contains('rollout-live')) return;
    playBtn.hidden = true;
    const { startPlay } = await import('./interact.js');
    play = startPlay({
      fig, worker, agent, fail,
      setStatus: (t) => { status.textContent = t; },
      onExit: ({ note }) => {
        play = null;
        caption(note);
        playBtn.hidden = !worker;
      },
    });
  };

  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onerror = fail;
  worker.addEventListener('message', ({ data: m }) => {
    if (m.type === 'ready') {
      fig.dataset.backend = m.backend;
      if (m.params) params = `${(m.params / 1e6).toFixed(1)}M-param `;
      run();
    }
    else if (m.type === 'frame' && m.seed === seed) {
      const c = canvases[m.i];
      next[m.i] = m.mask;
      paint(c, m.mask, colors);
      if (!c.classList.contains('live')) c.classList.add('live');
      else if (!calm) c.animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 350, easing: 'ease-out' });
    } else if (m.type === 'done' && m.seed === seed) {
      clearTimeout(timer);
      shown = next;
      shownSeed = seed;
      agent = m.agent;
      fig.dataset.ms = Math.round(m.ms);
      fig.classList.add('rollout-live', 'rollout-has-live');
      caption();
      again.hidden = playBtn.hidden = false;
      again.disabled = false;
    } else if (m.type === 'iframe' && m.seed === seed) {
      // Interactive frame: scroll the strip left by one.
      shown.shift();
      shown.push(m.mask);
      agent = m.agent;
      repaint();
      setLabels(m.n);
      if (!calm) canvases[canvases.length - 1].animate([{ opacity: 0.4 }, { opacity: 1 }], { duration: 250 });
    } else if (m.type === 'error') fail();
  });
  worker.postMessage({
    type: 'init', manifest: fig.dataset.manifest, ortBase: fig.dataset.ort, backends: list,
    allowFallback: fig.dataset.allowFallback === '1',
  });
  timer = setTimeout(fail, TIMEOUT);
  again.addEventListener('click', () => worker && run());
  playBtn.addEventListener('click', enterPlay);
  fig.querySelector('.rollout-frames').addEventListener('click', enterPlay);
  addEventListener('pagehide', () => { clearTimeout(timer); if (worker) worker.terminate(); }, { once: true });
  new MutationObserver(() => {
    colors = themeColors();
    canvases.forEach((c, i) => { const m = next[i] || shown[i]; if (m && c.classList.contains('live')) paint(c, m, colors); });
  }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] });
}

if (fig) {
  loaded().then(idle).then(() => onScreen(fig)).then(shown).then(() => {
    // data-backends="webgpu" on the figure turns the ORT-wasm fallback off.
    const allow = fig.dataset.backends ? fig.dataset.backends.split(',') : null;
    const list = backends().filter((b) => !allow || allow.includes(b));
    if (list.length) boot(list);
  });
}
