// Weight file format: 'TWM1', u32 header length, JSON header {cfg, tensors: {name: [offset, shape]}},
// then fp16 data. Decoded to Float32Array per tensor.
const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function parseWeights(buf) {
  const u8 = new Uint8Array(buf);
  if (String.fromCharCode(...u8.subarray(0, 4)) !== 'TWM1') throw new Error('bad weights');
  const hl = new DataView(buf).getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hl)));
  const half = new Uint16Array(buf, 8 + hl);
  const get = (name) => {
    const [off, shape] = head.tensors[name];
    const n = shape.reduce((a, b) => a * b, 1), out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = F16[half[off + i]];
    return out;
  };
  return { cfg: head.cfg, get };
}
