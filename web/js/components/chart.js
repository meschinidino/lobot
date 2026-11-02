import { stroke } from "../screens/evidence.js";
import { $, esc, fd } from "../state.js";
function chartScale(P, w, h, pad = 40) {
  const ys = (P.points || []).map((q) => q.trait), lo = Math.min(0, ...ys), hi = Math.max(1, ...ys);
  return { x: (c) => pad + Math.max(0, Math.min(1, c)) * (w - pad * 1.5), y: (t) => h - pad - ((t - lo) / (hi - lo)) * (h - pad * 1.6), lo, hi };
}
function chartPoints(P, w, h, pad = 40) {
  const sc = chartScale(P, w, h, pad);
  return (P.points || []).map((q) => ({ ...q, x: sc.x(q.cost), y: sc.y(q.trait), front: (P.front || []).includes(q.name) }));
}
// Greedy label placement. Marks are obstacles too. Try right of the mark, then left, then 15px steps up and down.
function placeLabels(items, w) {
  const boxes = items.map((it) => ({ x: it.x - 9, y: it.y + 9, w: 18, h: 18 }));
  const hit = (x, y, tw) => boxes.some((b) => x < b.x + b.w && b.x < x + tw && y - 12 < b.y && b.y - b.h < y + 2);
  return items.map((it) => {
    const tw = it.text.length * 7.3 + 4; let best = null;
    for (const dy of [0, -15, 15, -30, 30, -45, 45, 60, 75]) for (const x of [it.x + 12, it.x - 12 - tw]) {
      const y = it.y - 4 + dy;
      if (!best && x > 42 && x + tw < w - 4 && y > 26 && !hit(x, y, tw)) best = { x, y };
    }
    best = best || { x: it.x + 12, y: it.y - 6 };
    boxes.push({ x: best.x, y: best.y + 2, w: tw, h: 14 });
    return { ...it, lx: best.x, ly: best.y };
  });
}
function paretoChart(P, chosen, sel = chosen) {
  const W = 640, H = 300, cp = chartPoints(P, W, H), vec = cp.filter((q) => q.kind === "vector").sort((a, b) => a.dose - b.dose);
  const mark = (q) => { const f = q.front ? "currentColor" : "var(--sheet)";
    if (q.kind === "vector") return `<circle cx="${q.x}" cy="${q.y}" r="6" fill="${f}" stroke="currentColor" stroke-width="2"/>`;
    if (q.kind === "prompt") return `<rect x="${q.x - 6}" y="${q.y - 6}" width="12" height="12" fill="${f}" stroke="currentColor" stroke-width="2"/>`;
    return `<path d="M${q.x} ${q.y - 8}L${q.x + 8} ${q.y}L${q.x} ${q.y + 8}L${q.x - 8} ${q.y}Z" fill="${f}" stroke="currentColor" stroke-width="2"/>`; };
  const col = (q) => (q.kind === "vector" ? "var(--pen)" : q.kind === "prompt" ? "var(--pen-4)" : "var(--pen-3)");
  const lab = (q) => (q.kind === "vector" ? fd(q.dose) + (chosen != null && Math.abs(q.dose - chosen) < 1e-9 ? " chosen" : "") : q.name === "prompt:label" ? "label prompt" : q.name === "prompt" ? "drafted prompt" : "prompt + patch");
  const ch = cp.find((q) => q.kind === "vector" && sel != null && Math.abs(q.dose - sel) < 1e-9), sc = chartScale(P, W, H);
  const labs = placeLabels([...cp].sort((a, b) => a.y - b.y || a.x - b.x).map((q) => ({ x: q.x, y: q.y, text: lab(q), color: col(q) })), W);
  const yt = [0, 0.5, 1].filter((t) => t >= sc.lo && t <= sc.hi);
  return `<svg class="pchart" viewBox="0 0 ${W} ${H}" aria-hidden="true">
    <line x1="40" y1="${H - 40}" x2="${W - 20}" y2="${H - 40}" stroke="var(--ink)" stroke-width="2"/><line x1="40" y1="24" x2="40" y2="${H - 40}" stroke="var(--ink)" stroke-width="2"/>
    ${[0, 0.25, 0.5, 0.75, 1].map((t) => { const x = 40 + t * (W - 60); return `<line x1="${x}" y1="${H - 40}" x2="${x}" y2="${H - 34}" stroke="var(--ink)"/><text x="${x}" y="${H - 20}" text-anchor="middle" class="ax">${t.toFixed(2)}</text>`; }).join("")}
    <text x="${W - 20}" y="${H - 4}" text-anchor="end" class="ax">side-effect cost →</text><text x="46" y="16" class="ax">↑ effect</text>
    <text x="${W - 20}" y="16" text-anchor="end" class="ax muted">better: up and to the left</text>
    <polyline points="${vec.map((q) => `${q.x},${q.y}`).join(" ")}" fill="none" stroke="var(--pen)" stroke-width="2"/>
    ${yt.map((t) => `<line x1="34" y1="${sc.y(t)}" x2="40" y2="${sc.y(t)}" stroke="var(--ink)"/><text x="30" y="${sc.y(t) + 4}" text-anchor="end" class="ax">${t.toFixed(1)}</text>`).join("")}
    ${cp.map((q) => `<g style="color:${col(q)}">${mark(q)}</g>`).join("")}
    ${labs.map((l) => { const moved = Math.abs(l.ly - (l.y - 4)) > 1, left = l.lx < l.x, ax = left ? l.lx + l.text.length * 7.3 + 2 : l.lx - 2;
      return `${moved ? `<line x1="${l.x}" y1="${l.y}" x2="${ax}" y2="${l.ly - 4}" stroke="${l.color}" stroke-width="1"/>` : ""}<text x="${l.lx}" y="${l.ly}" class="ax" style="fill:${l.color}">${esc(l.text)}</text>`; }).join("")}
    ${ch ? `<circle cx="${ch.x}" cy="${ch.y}" r="12" fill="none" stroke="var(--ink)" stroke-width="3"/>` : ""}
  </svg>`;
}

export { chartScale, chartPoints, placeLabels, paretoChart };
