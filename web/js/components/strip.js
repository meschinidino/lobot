import { stroke } from "../screens/evidence.js";
import { $, at } from "../state.js";
function strip(el, box, draw) {
  const ws = [...el.querySelectorAll(".w[data-v]")]; if (!ws.length) return "";
  const H = 10, lines = [];
  for (const w of ws) { const b = w.getBoundingClientRect(), top = Math.round(b.top - box.top); let L = lines.at(-1); if (!L || Math.abs(L.top - top) > 4) lines.push(L = { top, y0: b.bottom - box.top + 5, segs: [] }); L.segs.push([b.left - box.left, b.right - box.left, +w.dataset.v]); }
  let out = "", onset = false;
  for (const L of lines) {
    const y = (v) => L.y0 + H - Math.max(-0.3, Math.min(1.6, v)) * H, x0 = L.segs[0][0], x1 = L.segs.at(-1)[1];
    out += `<rect x="${x0}" y="${L.y0}" width="${x1 - x0}" height="${H}" fill="var(--band)"/>`;
    let d = "";
    L.segs.forEach(([a, b, v], i) => { d += i ? ` V ${y(v)} H ${b}` : `M ${a} ${y(v)} H ${b}`; if (i < L.segs.length - 1) d += ` H ${L.segs[i + 1][0]}`;
      if (!onset && v >= 1) { onset = true; out += `<line x1="${a}" y1="${L.y0 - 6}" x2="${a}" y2="${L.y0 + H + 4}" stroke="var(--ink)" stroke-width="3"/>`; } });
    out += `<path class="${draw ? "draw" : ""}" d="${d}" fill="none" stroke="var(--pen)" stroke-width="2"/>`;
  }
  return out;
}

export { strip };
