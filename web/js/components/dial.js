import { update } from "../screens/workbench.js";
import { stroke } from "../screens/evidence.js";
import { $, S, eid, fd, parseDose, reduce } from "../state.js";
/* ---------- knob: turns the dose of the edit selected in the sentence ---------- */
const K = (() => {
  const host = $("knob"), C = 110, R = 96, A0 = -135, A1 = 135;
  const toA = (d) => A0 + (A1 - A0) * (d - S.limits.dose_min) / (S.limits.dose_max - S.limits.dose_min);
  const pt = (a, r) => [C + r * Math.sin(a * Math.PI / 180), C - r * Math.cos(a * Math.PI / 180)];
  function draw() {
    const L = S.limits, steps = Math.round((L.dose_max - L.dose_min) / 0.5);
    let ticks = "";
    for (let j = 0; j <= steps; j++) {
      const d = L.dose_min + j * 0.5, a = toA(d), major = Number.isInteger(d), [x1, y1] = pt(a, R - 2), [x2, y2] = pt(a, R - (d === 0 ? 20 : major ? 14 : 7));
      ticks += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${major ? "var(--ink)" : "var(--ink-muted)"}" stroke-width="${d === 0 ? 4 : major ? 2.5 : 1}"/>`;
      if (major) { const [tx, ty] = pt(a, R + 15); ticks += `<text x="${tx}" y="${ty + 4}" text-anchor="middle" font-family="JetBrains Mono" font-size="11" fill="var(--ink-muted)">${d > 0 ? "+" + d : d < 0 ? "−" + -d : "0"}</text>`; }
    }
    host.innerHTML = `<svg viewBox="-20 -20 260 260" role="slider" tabindex="0" aria-label="Dose of the selected edit" aria-valuemin="${L.dose_min}" aria-valuemax="${L.dose_max}"><circle cx="${C}" cy="${C}" r="${R}" fill="none" stroke="var(--ink)" stroke-width="2"/>${ticks}<circle cx="${C}" cy="${C}" r="62" fill="var(--sheet)" stroke="var(--ink)" stroke-width="2"/><rect class="ptr" x="${C - 4}" y="${C - 60}" width="8" height="36" fill="var(--ink)"/></svg>`;
    wire(host.querySelector("svg"));
  }
  function wire(svg) {
    const set = (d) => { const r = S.rows[S.focusEdit]; if (!r || S.running) return; d = Math.round(Math.max(S.limits.dose_min, Math.min(S.limits.dose_max, d)) * 20) / 20; if (d !== r.dose) { r.dose = d; update(); } };
    const from = (ev) => { const b = svg.getBoundingClientRect(); let a = Math.atan2(ev.clientX - (b.left + b.width / 2), -(ev.clientY - (b.top + b.height / 2))) * 180 / Math.PI; a = Math.max(A0, Math.min(A1, a)); set(S.limits.dose_min + (a - A0) / (A1 - A0) * (S.limits.dose_max - S.limits.dose_min)); };
    let drag = false;
    svg.addEventListener("pointerdown", (e) => { if (!S.rows.length || S.running) return; drag = true; svg.setPointerCapture(e.pointerId); from(e); });
    svg.addEventListener("pointermove", (e) => { if (drag) from(e); });
    svg.addEventListener("pointerup", () => (drag = false));
    // Fine steps on the arrows, coarse on Page Up and Page Down, ends on Home and End.
    svg.addEventListener("keydown", (e) => {
      const r = S.rows[S.focusEdit]; if (!r) return;
      const step = { ArrowRight: .05, ArrowUp: .05, ArrowLeft: -.05, ArrowDown: -.05, PageUp: .5, PageDown: -.5 }[e.key];
      if (step) { e.preventDefault(); set(r.dose + step); } else if (e.key === "Home") { e.preventDefault(); set(S.limits.dose_min); } else if (e.key === "End") { e.preventDefault(); set(S.limits.dose_max); }
    });
  }
  function show() {
    const svg = host.querySelector("svg"), r = S.rows[S.focusEdit], d = r ? r.dose : 0;
    const ptr = host.querySelector(".ptr"); ptr.style.transition = reduce ? "none" : "transform 160ms cubic-bezier(.65,0,.35,1)";
    ptr.setAttribute("transform", `rotate(${toA(d)} ${C} ${C})`); svg.style.opacity = r && !S.running ? 1 : .45;
    svg.setAttribute("aria-valuenow", d); svg.setAttribute("aria-valuetext", r ? `${eid(r)} ${fd(d)}` : "no edit selected");
  }
  return { draw, show };
})();
function renderKnob() {
  const r = S.rows[S.focusEdit];
  K.show();
  $("bigdose").textContent = r ? fd(r.dose) : "±0.00"; $("bigdose").disabled = !r || S.running;
  $("knobFor").textContent = r ? `dose of ${eid(r)} · arrows step 0.05, Page Up/Down 0.5` : "no edit · add one in the sentence";
}
$("bigdose").addEventListener("click", () => { const r = S.rows[S.focusEdit]; if (!r) return; $("bigdose").hidden = true; const i = $("doseIn"); i.hidden = false; i.value = fd(r.dose); i.focus(); i.select(); });
function commitDose(save) {
  const i = $("doseIn"), r = S.rows[S.focusEdit], v = parseDose(i.value);
  if (save && r && !Number.isNaN(v) && v >= S.limits.dose_min && v <= S.limits.dose_max) r.dose = v;
  i.hidden = true; $("bigdose").hidden = false; update(); $("bigdose").focus();
}
$("doseIn").addEventListener("keydown", (e) => { if (e.key === "Enter") commitDose(true); else if (e.key === "Escape") { e.stopPropagation(); commitDose(false); } });
$("doseIn").addEventListener("blur", () => { if (!$("doseIn").hidden) commitDose(true); });

export { K, renderKnob, commitDose };
