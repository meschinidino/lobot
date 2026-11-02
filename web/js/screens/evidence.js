import { update } from "./workbench.js";
import { creating, run, stop } from "./run.js";
import { behaviors, pop } from "./workbench.js";
import { $, S, at, esc, fd, reduce } from "../state.js";
/* ---------- pages: New behavior, Create progress, behavior evidence ---------- */
const short = (m) => (m || "").split("/").pop();
const num = (x, d = 2) => (x == null ? "n/a" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d));
const ci = (c) => (c ? `[${num(c[0])}, ${num(c[1])}]` : "");
const mins = (sec) => { const t = Math.round(sec); return `${Math.floor(t / 60)} min ${String(t % 60).padStart(2, "0")} s`; };
const STROKE = {
  read: `<line x1="0" y1="6" x2="48" y2="6" stroke="var(--ink)" stroke-width="2"/>`,
  update: `<line x1="0" y1="3" x2="48" y2="3" stroke="var(--ink-muted)" stroke-width="1.5"/><line x1="0" y1="9" x2="48" y2="9" stroke="var(--ink)" stroke-width="3"/>`,
  delete: `<line x1="0" y1="6" x2="14" y2="6" stroke="var(--ink)" stroke-width="2"/><line x1="34" y1="6" x2="48" y2="6" stroke="var(--ink)" stroke-width="2"/><path d="M14 1 V11 M34 1 V11 M18 11 L22 1 M24 11 L28 1 M30 11 L33 4" stroke="var(--ink)" stroke-width="1.5" fill="none"/>`,
};
const stroke = (k) => `<svg class="stroke" viewBox="0 0 48 12" aria-hidden="true">${STROKE[k]}</svg>`;
const RULES = { trait_ci_excludes_zero: "the effect's interval includes 0", trait_mean_ge_min: "the mean effect is under 0.30", control_upper_lt_max: "the random-direction control moved it too", paired_ci_excludes_zero: "it is not clearly above the control" };
const reports = {};
/* ---------- evidence v2 (U4): prompt comparison, dose rule, every guard screen, repair ---------- */
const vecPoint = (P, d) => (P && d != null ? (P.points || []).find((q) => q.kind === "vector" && Math.abs(q.dose - d) < 1e-9) : null);
const VERDICT = {
  vector_dominates: ["ok", "KNOB BETTER"], prompt_dominates: ["warn", "PROMPT BETTER"], tradeoff: ["info", "NO CLEAR WINNER"],
};
const pointName = (q) => (q.kind === "vector" ? `patch at ${fd(q.dose)}` : q.name === "prompt:label" ? "prompt from the judge label" : q.name === "prompt+vector" ? `prompt + patch at ${fd(q.dose)}` : "drafted prompt");
// One plain sentence for the PP verdict, built from the measured points only.
function paretoSentence(P) {
  const pr = (P.points || []).find((q) => q.name === P.verdict_prompt), bv = P.best_vector && (P.points || []).find((q) => q.name === P.best_vector.name);
  if (!pr) return "The prompt points were not measured.";
  if (P.verdict === "prompt_dominates") return `A plain prompt gives more effect with fewer side effects than the patch at every dose (${pointName(pr)}: effect ${num(pr.trait)}, cost ${pr.cost.toFixed(2)}).`;
  if (P.verdict === "vector_dominates") return bv ? `The patch at ${fd(bv.dose)} gives more effect (${num(bv.trait)} vs ${num(pr.trait)}) with fewer side effects (cost ${bv.cost.toFixed(2)} vs ${pr.cost.toFixed(2)}) than the ${pointName(pr)}.` : "The patch beats the prompt on both effect and cost.";
  return bv ? `Neither wins on both. The ${pointName(pr)} gives more effect (${num(pr.trait)} vs ${num(bv.trait)}); the patch at ${fd(bv.dose)} costs less (${bv.cost.toFixed(2)} vs ${pr.cost.toFixed(2)}).` : "Neither wins on both effect and cost.";
}
// Chart geometry: x = side-effect cost 0..1, y = effect (trait gain). Pure, so it is easy to check.




const followupKey = (st) => (!st ? null : st.status === "running" || st.status === "queued" ? ["info", st.status === "queued" ? "QUEUED" : `RUNNING ${st.done || 0}/${st.total || "?"}`] : st.status === "failed" ? ["err", "FAILED"] : st.status === "stopped" ? ["warn", "STOPPED"] : null);
const noWeights = () => (S.state?.replay ? "Replay mode loads no weights." : S.state?.no_model ? "This server runs without weights." : "");
function followupCtl(kind, id, R, label, est) {
  const st = R.followups?.[kind], k = followupKey(st), live = st && (st.status === "running" || st.status === "queued"), off = noWeights();
  return `<div class="row2">${live ? `<button class="btn" type="button" data-fu-stop="${kind}" data-id="${esc(id)}">Stop</button>` : `<button class="btn" type="button" data-fu="${kind}" data-id="${esc(id)}" ${off || creating() || S.running ? "disabled" : ""}>${label}</button>`}
    ${k ? `<span class="state ${k[0]}">${k[1]}</span>` : ""}<span class="meta">${live ? "The server keeps working if you close this page." : off ? `${off} Start the server with weights to run it.` : creating() ? "Wait for the running Create." : est}</span></div>
    ${st?.status === "failed" ? `<p class="meta" style="color:var(--error)">${esc((st.error || "").split("\n").filter(Boolean).pop() || "no reason given")}</p>` : ""}`;
}
function compareSection(R, id, fam) {
  const P = R.pareto, base = R.prompt_baseline || {};
  const head = `<h2 class="sec">Compared with a plain prompt</h2>`;
  const drafted = base.card ? `<span>Drafted prompt</span><span><span class="mono">${num(base.card.trait_mean)} ${ci(base.card.trait_ci)}</span> with “${esc(base.top || "")}”${looksLikeReply(base.top) ? ` <span class="meta">this baseline is a sample line, not an instruction</span>` : ""}</span>` : "";
  if (!P) return `${head}<p class="lead">This run has no prompt comparison yet. It scores the patch at five doses and two plain prompts on the same held-out questions, for effect and for side effects (relevance and fluency).</p>
    <div class="kv2">${drafted}</div>${followupCtl("pareto", id, R, "Compare with a prompt", "About 2 min on the llama lane, about 8 min on torch.")}`;
  const [kind, key] = VERDICT[P.verdict] || ["info", "NO VERDICT"];
  const lp = (P.points || []).find((q) => q.name === "prompt:label");
  const rows = (P.points || []).map((q) => `<tr${(P.front || []).includes(q.name) ? ' class="front"' : ""}><td>${esc(pointName(q))}${(P.front || []).includes(q.name) ? ' <span class="meta">front</span>' : ""}</td><td class="mono">${num(q.trait)}</td><td class="mono">${q.cost.toFixed(2)}</td><td class="mono">${q.relevance.toFixed(2)}</td><td class="mono">${q.fluency.toFixed(2)}</td></tr>`).join("");
  return `${head}<p class="verdict"><span class="state ${kind}">${key}</span> ${esc(paretoSentence(P))}</p>
    <figure class="pfig"><div id="pchartBox">${paretoChart(P, R.frozen?.dose, EXS[id]?.d ?? R.frozen?.dose)}</div><figcaption class="meta">Circles: the patch at each dose, joined in dose order. Squares: plain prompts. Diamond: prompt and patch together. Filled marks are on the front (nothing beats them on both). The ring follows the dial; "chosen" marks the dose Create picked. ${P.n} held-out questions, judge “${esc(P.labels?.trait || fam?.judge || "")}”.</figcaption></figure>
    <div class="evwrap"><table class="ev pt"><caption class="meta">Effect is the trait gain over the unedited reply (base trait ${P.base.trait.toFixed(2)}). Cost is the mean drop in relevance and fluency (base ${P.base.relevance.toFixed(2)} and ${P.base.fluency.toFixed(2)}).</caption><thead><tr><th scope="col">Point</th><th scope="col">Effect</th><th scope="col">Cost</th><th scope="col">Relevance</th><th scope="col">Fluency</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="kv2">${drafted}${lp ? `<span>Label prompt</span><span>“${esc(lp.instruction || "")}”</span>` : ""}</div>`;
}
// A drafted baseline that reads like a reply (a greeting or question to the user) is not an instruction.
const looksLikeReply = (s) => !!s && (/^(howdy|hello|hi|hey|ahoy|well|ah|oh|greetings)\b/i.test(s.trim()) || /\?\s*$/.test(s.trim()) && !/^(answer|reply|respond|speak|talk|write|use|describe)\b/i.test(s.trim()));
function doseRuleLine(R) {
  const rule = R.tuning?.rule, c = R.tuning?.chosen;
  if (rule) return `<p class="meta">Dose chosen by the fluency-aware rule (<span class="mono">${esc(rule.name)}</span>): NLL rise at most ${rule.nll_max_rise}, fluency at least ${rule.fluency_floor}.${c && c.fluency != null ? ` On tuning prompts the chosen dose had fluency <span class="mono">${c.fluency.toFixed(2)}</span>.` : ""}</p>`;
  const pt = vecPoint(R.pareto, R.frozen?.dose);
  return `<p class="note"><span class="state warn">CHOSEN BEFORE THE FLUENCY CHECK</span> ${pt ? (pt.fluency < 0.5 ? `At this dose the replies are not fluent (fluency <span class="mono">${pt.fluency.toFixed(2)}</span>). The trait judge can still score broken text high. Create a new version to get a fluent dose.` : `Fluency at this dose is <span class="mono">${pt.fluency.toFixed(2)}</span>, so the text reads well.`) : "This dose was chosen on NLL only, which passes repeated words. Compare with a prompt to see the text quality, or create a new version."}</p>`;
}
const SCREEN = { refusal: "Refusal", honesty: "Honesty", privacy: "Privacy", harmful: "Harmful requests" };
function screenMeasure(name, v) {
  const n = v.blocked_prompts?.length ?? 0;
  if (name === "refusal") return `${n} request${n === 1 ? "" : "s"} lost ${n === 1 ? "its" : "their"} refusal${v.min_ratio != null ? `; refusal kept at <span class="mono">${Math.round(v.min_ratio * 100)}%</span> of its usual level` : ""}`;
  if (name === "privacy") return `${n} bait question${n === 1 ? "" : "s"} leaked a planted secret${v.control_flips ? `; random directions leaked <span class="mono">${v.control_flips.join(", ")}</span>` : ""}`;
  return `${n} prompt${n === 1 ? "" : "s"} flipped${v.min_ratio != null ? ` (worst kept <span class="mono">${Math.round(v.min_ratio * 100)}%</span>)` : ""}`;
}
function guardSection(R) {
  const g = R.guard;
  if (!g) return `<h2 class="sec">Guard</h2><p class="meta">The guard was not recorded for this run.</p>`;
  const screens = g.screens || { refusal: g };
  const rows = Object.entries(screens).map(([n, v]) => `<li><div><b>${SCREEN[n] || esc(n)}</b><br><span class="meta">${screenMeasure(n, v)}${v.display_prompt && (v.blocked || v.would_block) ? ` · worst: ${esc(v.display_prompt)}` : ""}</span></div>
    <span class="state ${v.blocked ? "err" : v.would_block ? "warn" : "ok"}">${v.blocked ? "BLOCKED" : v.would_block ? "WOULD BLOCK" : "PASSED"}</span></li>`).join("");
  return `<h2 class="sec">Guard</h2><p class="meta">${g.verified ? "Confirmed by full generation." : "First-token check only; publishing needs a full-generation check."}${g.enforced ? " Enforced in this Create." : " Recorded, not enforced, in this Create."}${g.research_mode ? " Research mode." : ""}</p><ol class="ruled screens">${rows}</ol>`;
}
/* ---------- 100x: evidence you can turn ---------- */
// Turn the dose: a printed dial with one detent per recorded dose. It shows recorded replies only; it never generates.
const EXS = {}; // per job: { d: selected dose, qi: question index }
const exDoses = (P) => [0, ...(P.points || []).filter((q) => q.kind === "vector").map((q) => q.dose).sort((a, b) => a - b)];
const exQuestions = (P) => (Object.values(P.samples || {})[0] || []).map((x) => x.q);
const sampleFor = (P, name, q) => (P.samples?.[name] || []).find((x) => x.q === q)?.reply ?? null;
function baseFor(R, q) {
  for (const v of ["update_up", "update_down", "delete"]) { const p = (R.final?.[v]?.pairs || []).find((x) => x.q === q); if (p && p.base) return p.base; }
  return null;
}
// Index of the first word where text leaves base (null when they match to the end).
function splitWord(base, text) {
  const A = String(base).trim().split(/\s+/), B = String(text).trim().split(/\s+/);
  let i = 0; while (i < A.length && i < B.length && A[i] === B[i]) i++;
  return i < B.length ? i : null;
}
function markSplit(text, base, cls) {
  const i = base == null ? null : splitWord(base, text); let w = -1;
  return String(text).trim().split(/(\s+)/).map((t) => { if (!t || /^\s+$/.test(t)) return esc(t); w++; return w === i ? `<mark class="${cls}">${esc(t)}</mark>` : esc(t); }).join("");
}
function exState(R, id) {
  const P = R.pareto, ds = exDoses(P);
  if (!EXS[id]) EXS[id] = { d: ds.find((d) => Math.abs(d - (R.frozen?.dose ?? -1)) < 1e-9) ?? ds[1] ?? 0, qi: 0 };
  return EXS[id];
}
const DIAL = { C: 100, R: 80, A0: -135, A1: 135 };
const dialAngle = (i, n) => DIAL.A0 + (DIAL.A1 - DIAL.A0) * (n > 1 ? i / (n - 1) : 0.5);
function exDial(ds, d) {
  const { C, R } = DIAL, pt = (a, r) => [C + r * Math.sin(a * Math.PI / 180), C - r * Math.cos(a * Math.PI / 180)];
  const ticks = ds.map((x, i) => { const a = dialAngle(i, ds.length), [x1, y1] = pt(a, R - 2), [x2, y2] = pt(a, R - (x === 0 ? 20 : 13)), [tx, ty] = pt(a, R + 16);
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--ink)" stroke-width="${x === 0 ? 4 : 2.5}"/><text x="${tx}" y="${ty + 4}" text-anchor="middle" class="dl">${x === 0 ? "0" : fd(x)}</text>`; }).join("");
  const i = Math.max(0, ds.findIndex((x) => Math.abs(x - d) < 1e-9));
  return `<svg id="exDial" viewBox="-14 -14 228 228" role="slider" tabindex="0" aria-label="Dose to show" aria-valuemin="${ds[0]}" aria-valuemax="${ds[ds.length - 1]}" aria-valuenow="${d}" aria-valuetext="${d === 0 ? "no edit" : `dose ${fd(d)}`}">
    <circle cx="${C}" cy="${C}" r="${R}" fill="none" stroke="var(--ink)" stroke-width="2"/>${ticks}
    <circle cx="${C}" cy="${C}" r="50" fill="var(--sheet)" stroke="var(--ink)" stroke-width="2"/>
    <rect class="ptr" x="${C - 4}" y="${C - 48}" width="8" height="30" fill="var(--ink)" transform="rotate(${dialAngle(i, ds.length)} ${C} ${C})"/></svg>`;
}
function exOut(R, id) {
  const P = R.pareto, st = exState(R, id), qs = exQuestions(P), q = qs[st.qi] ?? qs[0], d = st.d, pt = d === 0 ? null : vecPoint(P, d), chosen = R.frozen?.dose;
  const [k, key] = d === 0 ? ["info", "NO EDIT"] : !pt ? ["info", "NOT RECORDED"] : pt.fluency < 0.5 ? ["warn", "NOT FLUENT"] : chosen != null && Math.abs(d - chosen) < 1e-9 ? ["ok", "CREATE'S DOSE"] : ["info", "TESTED DOSE"];
  const nums = d === 0 ? `effect ${num(0)} · cost 0.00 · relevance ${P.base.relevance.toFixed(2)} · fluency ${P.base.fluency.toFixed(2)}`
    : pt ? `effect ${num(pt.trait)} · cost ${pt.cost.toFixed(2)} · relevance ${pt.relevance.toFixed(2)} · fluency ${pt.fluency.toFixed(2)}` : "not recorded";
  const base = q != null ? baseFor(R, q) : null, patch = d === 0 || !pt ? null : sampleFor(P, pt.name, q);
  const pn = P.verdict_prompt || "prompt:label", pr = q != null ? sampleFor(P, pn, q) : null, prPoint = (P.points || []).find((x) => x.name === pn);
  const branch = (label, text, html, note) => `<div class="br"><div class="bl mono">${label}${note ? ` <span class="meta">${note}</span>` : ""}</div><p class="bt">${text == null ? `<span class="meta">not recorded for this question</span>` : html}</p></div>`;
  return `<div class="exread"><span class="exbig">${d === 0 ? "±0.00" : fd(d)}</span><div><span class="state ${k}">${key}</span><p class="mono exnums">${nums}</p>
      ${d !== 0 && pt && pt.fluency < 0.5 ? `<p class="meta">The judge scores this text as not fluent. A trait score here can come from broken words.</p>` : ""}</div></div>
    ${qs.length > 1 ? `<div class="row2" role="group" aria-label="Question">${qs.map((x, i) => `<button class="btn btn-sm" type="button" data-exq="${i}" aria-pressed="${i === st.qi}">Question ${i + 1}</button>`).join("")}</div>` : ""}
    <p class="exq">“${esc(q ?? "")}”</p>
    <div class="fork">
      ${branch("no edit", base, esc(base ?? ""), "")}
      ${d === 0 ? "" : branch(`patch ${fd(d)}`, patch, patch == null ? "" : markSplit(patch, base, "cut"), "first changed word in pink")}
      ${branch(prPoint ? pointName(prPoint) : "plain prompt", pr, pr == null ? "" : markSplit(pr, base, "ink"), prPoint ? `effect ${num(prPoint.trait)} · cost ${prPoint.cost.toFixed(2)}` : "")}
    </div>
    <p class="meta">Recorded replies from the prompt comparison: ${P.n} held-out questions were scored; ${qs.length} are kept as samples. Greedy decoding, same seed.</p>`;
}
function setExDose(id, d) {
  const R = reports[id]; if (!R?.pareto) return;
  const ds = exDoses(R.pareto), st = exState(R, id); st.d = d;
  const svg = $("exDial"), i = Math.max(0, ds.findIndex((x) => Math.abs(x - d) < 1e-9));
  if (svg) { const ptr = svg.querySelector(".ptr"); ptr.style.transition = reduce ? "none" : "transform 160ms cubic-bezier(.65,0,.35,1)"; ptr.setAttribute("transform", `rotate(${dialAngle(i, ds.length)} ${DIAL.C} ${DIAL.C})`);
    svg.setAttribute("aria-valuenow", d); svg.setAttribute("aria-valuetext", d === 0 ? "no edit" : `dose ${fd(d)}`); }
  if ($("exOut")) $("exOut").innerHTML = exOut(R, id);
  if ($("pchartBox")) $("pchartBox").innerHTML = paretoChart(R.pareto, R.frozen?.dose, d);
}
function wireExplorer(id) {
  const svg = $("exDial"), R = reports[id]; if (!svg || !R?.pareto) return;
  const ds = exDoses(R.pareto), idx = () => Math.max(0, ds.findIndex((x) => Math.abs(x - exState(R, id).d) < 1e-9));
  const fromPtr = (ev) => { const b = svg.getBoundingClientRect(); let a = Math.atan2(ev.clientX - (b.left + b.width / 2), -(ev.clientY - (b.top + b.height / 2))) * 180 / Math.PI;
    a = Math.max(DIAL.A0, Math.min(DIAL.A1, a)); const i = Math.round((a - DIAL.A0) / (DIAL.A1 - DIAL.A0) * (ds.length - 1)); if (ds[i] !== exState(R, id).d) setExDose(id, ds[i]); };
  let drag = false;
  svg.addEventListener("pointerdown", (e) => { drag = true; svg.setPointerCapture(e.pointerId); fromPtr(e); });
  svg.addEventListener("pointermove", (e) => { if (drag) fromPtr(e); });
  svg.addEventListener("pointerup", () => (drag = false));
  svg.addEventListener("keydown", (e) => {
    const i = idx(), j = { ArrowRight: i + 1, ArrowUp: i + 1, ArrowLeft: i - 1, ArrowDown: i - 1, Home: 0, End: ds.length - 1 }[e.key];
    if (j == null) return; e.preventDefault(); setExDose(id, ds[Math.max(0, Math.min(ds.length - 1, j))]);
  });
}
function explorerSection(R, id) {
  const P = R.pareto; if (!P || !P.samples) return "";
  const st = exState(R, id), ds = exDoses(P);
  return `<h2 class="sec">Turn the dose</h2><p class="lead">Each detent is a dose that was measured. Turn the dial to read the recorded replies at that dose, next to the reply with no edit and the reply to a plain prompt.</p>
    <div class="ex"><div class="exdial">${exDial(ds, st.d)}<span class="meta">arrows step one detent</span></div><div id="exOut" class="exout">${exOut(R, id)}</div></div>`;
}

// How Create chose the dose: the tuning grid with the rule's reason for each cell.
function tuningSection(R) {
  const T = R.tuning; if (!T?.table || !Object.keys(T.table).length) return "";
  const rule = T.rule, maxRise = rule?.nll_max_rise ?? 1, floor = rule?.fluency_floor ?? 0.5, fr = R.frozen || {};
  const layers = Object.keys(T.table), dkeys = [...new Set(layers.flatMap((L) => Object.keys(T.table[L])))].sort((a, b) => Number(a) - Number(b));
  const zero = dkeys.find((k) => Number(k) === 0), doses = dkeys.filter((k) => Number(k) > 0);
  const hasFl = layers.some((L) => Object.values(T.table[L]).some((c) => c.fluency != null));
  const cell = (L, k) => {
    const c = T.table[L][k]; if (!c) return `<td class="meta">–</td>`;
    const b = T.table[L][zero] || {}, rise = c.nll - (b.nll ?? 0), nllOk = rise <= maxRise, flOk = c.fluency == null || c.fluency >= floor;
    const chosen = Number(L) === fr.layer && Math.abs(Number(k) - (fr.dose ?? -1)) < 1e-9;
    const why = chosen ? `<span class="state ok">CHOSEN</span>` : !nllOk ? `<span class="state info">NLL TOO HIGH</span>` : rule && !flOk ? `<span class="state warn">NOT FLUENT</span>` : !rule && c.fluency != null && !flOk ? `<span class="state warn">NOT FLUENT</span>` : `<span class="state info">QUALIFIED</span>`;
    return `<td class="${chosen ? "chosen" : nllOk && flOk ? "" : "rej"}"><span class="mono">trait ${c.judge.toFixed(2)}</span><br><span class="mono">NLL ${num(rise)}</span>${c.fluency != null ? `<br><span class="mono">fluency ${c.fluency.toFixed(2)}</span>` : ""}<br>${why}</td>`;
  };
  const rows = layers.map((L) => `<tr><th scope="row" class="mono">layer ${esc(L)}</th>${doses.map((k) => cell(L, k)).join("")}</tr>`).join("");
  return `<h2 class="sec">How Create chose the dose</h2>
    <p class="lead">Create scored each candidate layer and dose on tuning prompts (never the held-out ones). ${rule ? `A cell qualifies when the reply NLL rises by at most ${maxRise} and the fluency score is at least ${floor}. The qualifying cell with the highest trait score wins.` : `This run used the older rule: a cell qualified when the reply NLL rose by at most ${maxRise}. NLL alone passes repeated words, so a broken dose could win.${hasFl ? "" : " Fluency was not measured in this run."}`}</p>
    <div class="evwrap"><table class="ev pt tgrid"><thead><tr><th scope="col">Layer</th>${doses.map((k) => `<th scope="col" class="mono">${fd(Number(k))}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

// In plain words: Muse Spark explains the report from its own numbers; the page keeps the numbers above as the source.
const explains = {};
function explainSection(id) {
  const x = explains[id];
  return `<h2 class="sec">In plain words</h2>${x?.text ? `<div class="explain"><p>${esc(x.text).replace(/\n\n+/g, "</p><p>").replace(/\n/g, "<br>")}</p></div><p class="meta">Written by Muse Spark from this report's numbers only${x.cached ? " (saved earlier)" : ""}. The numbers above are the source; check them there.</p>`
    : `<div class="row2"><button class="btn" type="button" data-explain="${esc(id)}" ${x?.busy ? "disabled" : ""}>${x?.busy ? "Writing…" : "Explain with Muse"}</button><span class="meta">${x?.err ? esc(x.err) : "Muse Spark reads only this report's numbers and writes a short explanation. It needs META_API_KEY on the server."}</span></div>`}`;
}

// What each state key means. One sentence each, shown in legends and as hover text.
const KEY_HELP = {
  "PASS": "Create found a dose that passed its checks, and no other key applies. Open the evidence for each verb.",
  "INCREASE FAIL": "Create found a dose, but turning the behavior up did not pass its held-out report card. You can still try it.",
  "DECREASE FAIL": "Turning the behavior up passed, but turning it down (the negative dose) did not pass its report card.",
  "GUARD": "A guard check would block this patch: for example, it weakened refusals of harmful requests. It is recorded, not enforced, so you can still use it with care. Red means the guard blocked it, and it cannot be used.",
  "BLOCKED": "The guard blocked this patch. Research mode lets you use it anyway; every run records that.",
  "NO KNOB": "Create found no layer and dose that passed its checks. Nothing was saved to use.",
  "FAIL": "The held-out report card failed. The reason is printed under it.",
  "NOT TESTED": "This verb was not measured in this run.",
  "KNOB BETTER": "The patch gives more effect with fewer side effects than a plain prompt.",
  "PROMPT BETTER": "A plain prompt gives more effect with fewer side effects than the patch at every dose.",
  "NO CLEAR WINNER": "Neither wins on both: one gives more effect, the other costs less.",
  "NOT FLUENT": "The judge scores the replies at this dose as not fluent text. A high trait score there can come from broken words.",
  "FLAGGED": "This guard check would block the patch.",
  "OK": "This guard check passed.",
};
const SHELF_KEYS = ["PASS", "INCREASE FAIL", "DECREASE FAIL", "GUARD", "NO KNOB"];

export { short, num, ci, mins, STROKE, stroke, RULES, reports, vecPoint, VERDICT, pointName, paretoSentence, followupKey, noWeights, followupCtl, compareSection, looksLikeReply, doseRuleLine, SCREEN, screenMeasure, guardSection, EXS, exDoses, exQuestions, sampleFor, baseFor, splitWord, markSplit, exState, DIAL, dialAngle, exDial, exOut, setExDose, wireExplorer, explorerSection, tuningSection, explains, explainSection, KEY_HELP, SHELF_KEYS };
