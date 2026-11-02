import { KEY_HELP, num, short, stroke } from "./evidence.js";
import { renderGuard, run, setStatus } from "./run.js";
import { loadBehaviors, openPage } from "./pages.js";
import { chatSend, drawChatStrips } from "./chat.js";
import { setView } from "./share.js";
import { $, EXPS, S, at, curExp, editsText, eid, esc, fd, isB, nid, parseDose, reduce, root, rowsKey, saveExps, store } from "../state.js";
import { postJSON } from "../api.js";
/* ---------- shelf ---------- */
function renderShelf() {
  const models = S.state ? S.state.models : [{ id: S.model || "Qwen/Qwen3.5-4B", short: (S.model || "Qwen3.5-4B").split("/").pop() }];
  $("models").innerHTML = models.map((m) => {
    const cur = m.id === S.model, st = !S.api ? "server offline" : S.state.replay ? "replay" : cur ? (S.state.no_model ? "no weights" : S.state.warm ? "warm" : "loads on run") : m.gated ? "gated" : "on disk";
    return `<button class="model" type="button" data-model="${esc(m.id)}" title="Switching reloads weights." aria-current="${cur}" ${!S.api || S.running ? 'aria-disabled="true"' : ""}><b>${esc(m.id.split("/").pop())}</b><span class="meta">${st}</span></button>`;
  }).join("");
  const list = Object.values(EXPS).filter((e) => e.model === S.model).sort((a, b) => b.at - a.at);
  $("expCount").textContent = list.length ? String(list.length) : "";
  $("exps").innerHTML = list.length ? list.map((e) => `<button class="exp" type="button" data-prompt="${esc(e.prompt)}" aria-current="${e.prompt === $("prompt").value.trim()}"><span class="p">${esc(e.prompt.length > 70 ? e.prompt.slice(0, 69) + "…" : e.prompt)}</span><span class="meta">${e.runs.length + 1} runs</span></button>`).join("")
    : `<p class="empty">No experiments for this model yet. Type a prompt and press Run.</p>`;
  $("railText").textContent = `${(S.model || "").split("/").pop()} · ${S.rows.length} edit${S.rows.length === 1 ? "" : "s"}`;
  renderBehaviors();
}

/* ---------- behaviors: saved traits (evidence API, else finished Create jobs) and running Creates ---------- */
const STAGES = [["extract", "Contrast prompts and capture"], ["tune", "Tune layer and dose"], ["read_gate", "Read check"], ["final", "Final check"], ["baseline", "Prompt baseline"], ["pareto", "Compare with a prompt"], ["guard", "Guard"], ["verdict", "Report card"]];
const NST = STAGES.length;
const active = (j) => j.status === "queued" || j.status === "running";
function behaviors() {
  const mine = S.jobs.filter((j) => j.model_id === S.model);
  const ver = {}, out = [];
  if (S.traits) for (const t of S.traits) out.push({ id: t.id, name: t.family, version: t.version, verdict: t.verdict, guard: t.guard || {}, cards: t.cards || {}, dose: t.dose, layer: t.layer, patch: t.patch, created: t.created, ev: t.evidence || {}, del: !!t.delete_eligible });
  else for (const j of mine.filter((j) => j.status === "done").sort((a, b) => a.created - b.created)) {
    ver[j.trait] = (ver[j.trait] || 0) + 1;
    out.push({ id: j.id, name: j.trait, version: ver[j.trait], verdict: j.verdict, guard: j.guard || {}, cards: {}, dose: j.frozen?.dose, layer: j.frozen?.layer, patch: null, created: j.created });
  }
  // Only the newest version of each family is offered in the sentence; older ones stay on the evidence page.
  const newest = {};
  for (const b of out) if (!newest[b.name] || b.version > newest[b.name].version) newest[b.name] = b;
  for (const b of out) {
    b.newest = newest[b.name] === b;
    [b.kind, b.key, b.usable] = b.verdict === "blocked" || b.guard.blocked
      ? (S.research ? ["warn", "BLOCKED", true] : ["err", "GUARD", false])
      : b.verdict !== "knob" ? ["warn", "NO KNOB", false]
      : b.guard.would_block ? ["warn", "GUARD", true]
      : b.cards.update_up && !b.cards.update_up.passed ? ["warn", "INCREASE FAIL", true]
      : b.cards.update_down && !b.cards.update_down.passed ? ["warn", "DECREASE FAIL", true] : ["ok", "PASS", true];
  }
  return { saved: out.filter((b) => b.newest).sort((a, b) => b.created - a.created), all: out, running: mine.filter(active) };
}
function renderBehaviors() {
  const B = behaviors(), cur = S.page && S.page.id;
  const run = B.running.map((j) => { const i = STAGES.findIndex(([k]) => k === j.stage) + 1;
    return `<div class="beh"><button class="open" type="button" data-job="${esc(j.id)}" aria-current="${cur === j.id}"><b>${esc(j.trait)}</b><span class="state info">${j.status === "queued" ? "QUEUED" : `CREATING ${i}/${NST}`}</span></button><span></span></div>`; });
  const saved = B.saved.map((b) => `<div class="beh${b.usable ? "" : " dim"}"><button class="open" type="button" data-beh="${esc(b.id)}" aria-current="${cur === b.id}" aria-label="Evidence for ${esc(b.name)} v${b.version}"><b>${esc(b.name)}</b><span class="meta">v${b.version}</span> <span class="state ${b.kind}" title="${esc(KEY_HELP[b.key] || "")}">${b.key}</span></button>
    <button class="btn btn-sm" type="button" data-use="${esc(b.id)}" ${!b.usable || S.running || S.rows.some((r) => r.trait === b.name) ? "disabled" : ""} aria-label="Use ${esc(b.name)} in the sentence">Use</button></div>`);
  const noCreate = S.state?.replay ? "Replay mode loads no weights, so Create is off." : S.state?.no_model ? "The server runs without weights, so Create is off." : "";
  $("behaviors").innerHTML = run.join("") + saved.join("") || (!S.api ? `<p class="empty">Behaviors load from the server.</p>`
    : `<p class="empty">No validated behavior for this model yet. ${noCreate || `<button class="link" type="button" data-new>Create one from a family</button>.`}</p>`);
  $("newBeh").disabled = !S.api || !!S.state?.replay || !!S.state?.no_model;
}
function useBehavior(id) {
  const b = behaviors().all.find((x) => x.id === id); if (!b || !b.usable) return;
  // A run uses neurons or one behavior: using a behavior replaces the edits, at its frozen dose.
  S.rows = [{ trait: b.name, dose: b.dose ?? 1 }]; S.focusEdit = 0; setView("work"); update();
  setStatus("info", "READY", `${b.name} is in the sentence at its tested dose ${fd(b.dose ?? 1)}. Press Run, or Run at 5 doses.`);
}
$("behaviors").addEventListener("click", (e) => {
  const t = e.target;
  if (t.closest("[data-use]")) useBehavior(t.closest("[data-use]").dataset.use);
  else if (t.closest("[data-beh]")) openPage({ kind: "behavior", id: t.closest("[data-beh]").dataset.beh });
  else if (t.closest("[data-job]")) openPage({ kind: "create", id: t.closest("[data-job]").dataset.job });
  else if (t.closest("[data-new]")) openPage({ kind: "new" });
});
$("newBeh").addEventListener("click", () => openPage({ kind: "new" }));
$("libBtn").addEventListener("click", () => openPage({ kind: "library" }));
$("chatBtn").addEventListener("click", () => openPage({ kind: "chat" }));
$("vPage").addEventListener("keydown", (e) => { if (e.target.id === "chatIn" && e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); chatSend(); } });
new ResizeObserver(() => { if (S.page?.kind === "chat") requestAnimationFrame(drawChatStrips); }).observe($("vPage"));
$("models").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-model]"); if (!b || b.getAttribute("aria-disabled") === "true" || b.dataset.model === S.model) return;
  b.querySelector(".meta").textContent = "switching";
  try {
    const r = await fetch("api/model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: b.dataset.model }) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
    applyState(body); await loadBehaviors();
    S.rows = body.preset_rows.map((x) => ({ ...x })); S.focusEdit = 0;
    setStatus("info", "SWITCHED", "Switching reloads weights. The next run loads them.");
    update();
  } catch (err) { setStatus("err", "ERROR", String(err.message || err)); renderShelf(); }
});
$("exps").addEventListener("click", (e) => { const b = e.target.closest("[data-prompt]"); if (!b || S.running) return; $("prompt").value = b.dataset.prompt; S.sel = null; update(); });

/* ---------- sentence ---------- */
function renderSentence() {
  const edits = S.rows.map((r, i) => `<span class="edit" aria-current="${i === S.focusEdit}">${i ? (i === S.rows.length - 1 ? " and " : ", ") : ""}<button class="link${isB(r) ? "" : " mono"}" type="button" data-focus="${i}" style="font-size:20px" aria-label="Select ${esc(eid(r))} for the knob${isB(r) ? "" : "; press again to edit"}">${esc(eid(r))}</button> at <span class="dose">${fd(r.dose)}</span><button class="x" type="button" data-remove="${i}" aria-label="Remove ${esc(eid(r))}" ${S.running ? "disabled" : ""}>×</button></span>`).join("");
  const full = S.rows.length >= S.limits.max_rows || S.rows.some(isB);
  $("sentence").innerHTML = `On <span class="mono" style="font-size:20px">${esc((S.model || "").split("/").pop())}</span>,
    ${S.rows.length ? `push ${edits}` : "with no edit"} <button class="btn btn-sm" type="button" id="addBtn" ${full || S.running ? "disabled" : ""} ${full ? 'title="A run uses up to 4 neurons, or one behavior."' : ""}>+ edit</button>
    ${S.read ? `and read <span class="edit"><button class="link" type="button" data-readsel style="font-size:20px">${esc(S.read)}</button><button class="x" type="button" data-unread aria-label="Stop reading ${esc(S.read)}" ${S.running ? "disabled" : ""}>×</button></span>` : readable().length ? `<button class="btn btn-sm" type="button" id="readBtn" ${S.running ? "disabled" : ""}>+ read</button>` : ""} and ask:`;
}
// "+ read": watch one saved behavior at every word (read dial). It works with neuron edits.
const readable = () => behaviors().saved.filter((b) => b.verdict === "knob");
function readMenu(anchor) {
  openPop(anchor, readable().map((b) => `<button class="opt" type="button" data-rd="${esc(b.name)}"><span><b>${esc(b.name)}</b> <span class="meta">v${b.version}</span></span><span class="meta">${S.read === b.name ? "reading" : "read"}</span></button>`).join(""));
  pop.addEventListener("click", (e) => { const t = e.target.closest("[data-rd]"); if (!t) return; S.read = t.dataset.rd; closePop(); update(); });
}
// "+ edit": push a neuron, or use a saved behavior. A run uses neurons or one behavior, not both.
function addMenu(anchor) {
  const B = behaviors().saved, hasN = S.rows.length > 0;
  const beh = B.map((b) => { const why = !b.usable ? b.key.toLowerCase() : hasN ? "remove the neurons first" : "";
    return `<button class="opt" type="button" data-addb="${esc(b.id)}" ${why ? 'aria-disabled="true"' : ""}><span><b>${esc(b.name)}</b> <span class="meta">v${b.version} · tested at ${fd(b.dose ?? 1)}</span></span><span class="meta">${why || "behavior"}</span></button>`; }).join("");
  openPop(anchor, `<button class="opt" type="button" data-addn><span><b>Push a neuron</b></span><span class="meta">layer, neuron, dose</span></button>${beh || `<button class="opt" type="button" data-addnew><span><b>New behavior</b></span><span class="meta">Create from a family</span></button>`}`);
  pop.addEventListener("click", (e) => {
    const t = e.target.closest(".opt"); if (!t || t.getAttribute("aria-disabled") === "true") return;
    if (t.hasAttribute("data-addn")) rowForm(anchor); else if (t.hasAttribute("data-addnew")) { closePop(); openPage({ kind: "new" }); } else { closePop(); useBehavior(t.dataset.addb); }
  });
}
let pop = null;
function closePop() { pop?.remove(); pop = null; }
function openPop(anchor, html) {
  closePop(); pop = document.createElement("div"); pop.className = "pop"; pop.innerHTML = html;
  const r = anchor.getBoundingClientRect(); pop.style.left = Math.max(8, Math.min(innerWidth - 340, r.left + scrollX)) + "px"; pop.style.top = r.bottom + scrollY + 6 + "px";
  document.body.append(pop); pop.querySelector("input, .opt")?.focus();
}
document.addEventListener("click", (e) => { if (pop && e.target.isConnected && !pop.contains(e.target) && !e.target.closest("#addBtn,#readBtn,[data-readsel],[data-focus]")) closePop(); });
// One form adds or edits a neuron row; it validates against the active model's limits.
function rowForm(anchor, i) {
  const r = i == null ? { layer: S.rows.at(-1)?.layer ?? 0, neuron: 0, dose: 1 } : S.rows[i], L = S.limits;
  openPop(anchor, `<form id="rowForm" novalidate><span class="label">${i == null ? "Push a neuron" : `Edit ${nid(r)}`}</span>
    <div class="grid3"><label><span class="meta">layer 0-${L.layers - 1}</span><input class="num" name="layer" inputmode="numeric" value="${r.layer}"></label>
    <label><span class="meta">neuron 0-${L.neurons - 1}</span><input class="num" name="neuron" inputmode="numeric" value="${r.neuron}"></label>
    <label><span class="meta">dose</span><input class="num" name="dose" inputmode="decimal" value="${fd(r.dose)}"></label></div>
    <p class="err" id="rowErr" role="alert"></p>
    <div class="row2"><button class="btn btn-sm" type="submit">${i == null ? "Add" : "Save"}</button><button class="btn btn-sm" type="button" id="rowCancel">Cancel</button></div></form>`);
  $("rowCancel").addEventListener("click", closePop);
  $("rowForm").addEventListener("submit", (e) => {
    e.preventDefault(); const f = e.target, v = { layer: Number(f.layer.value), neuron: Number(f.neuron.value), dose: parseDose(f.dose.value) };
    const bad = !Number.isInteger(v.layer) || v.layer < 0 || v.layer >= L.layers ? ["layer", `Layer must be 0 to ${L.layers - 1}.`]
      : !Number.isInteger(v.neuron) || v.neuron < 0 || v.neuron >= L.neurons ? ["neuron", `Neuron must be 0 to ${L.neurons - 1}.`]
      : Number.isNaN(v.dose) || v.dose < L.dose_min || v.dose > L.dose_max ? ["dose", `Dose must be ${L.dose_min} to +${L.dose_max}.`] : null;
    f.querySelectorAll(".num").forEach((x) => x.removeAttribute("aria-invalid"));
    if (bad) { f[bad[0]].setAttribute("aria-invalid", "true"); $("rowErr").textContent = bad[1]; f[bad[0]].focus(); return; }
    const dup = S.rows.findIndex((x, j) => j !== i && nid(x) === nid(v));
    if (dup >= 0) { $("rowErr").textContent = `${nid(v)} is already in the edits.`; return; }
    if (i == null) S.rows.push(v); else S.rows[i] = v;
    S.focusEdit = i == null ? S.rows.length - 1 : i; closePop(); update();
  });
}
$("sentence").addEventListener("click", (e) => {
  const t = e.target;
  if (t.closest("#addBtn")) addMenu(t.closest("#addBtn"));
  else if (t.closest("#readBtn") || t.closest("[data-readsel]")) readMenu(t.closest("#readBtn,[data-readsel]"));
  else if (t.closest("[data-unread]")) { S.read = null; update(); }
  else if (t.closest("[data-remove]")) { S.rows.splice(+t.closest("[data-remove]").dataset.remove, 1); S.focusEdit = Math.max(0, Math.min(S.focusEdit, S.rows.length - 1)); update(); }
  else if (t.closest("[data-focus]")) { const i = +t.closest("[data-focus]").dataset.focus; if (i === S.focusEdit && !S.running) { if (!isB(S.rows[i])) rowForm(t.closest("[data-focus]"), i); } else { S.focusEdit = i; update(); } }
});
$("prompt").addEventListener("input", () => { S.sel = null; renderShelf(); renderTree(); });

/* ---------- tree ---------- */
function treeRuns() {
  const e = curExp(); if (!e) return null;
  const runs = [{ n: 1, rows: [], text: e.baseline.text, split: null, dial: e.baseDial }, ...e.runs];
  const cs = runs.filter((r) => r.split).map((r) => r.split.char);
  let c = cs.length ? Math.min(...cs) : (/^\S+\s+\S+\s*/.exec(e.baseline.text) || [""])[0].length;
  // A split offset can point at the space before a word (BPE tokens carry it): step onto the word, then back to its start.
  while (c < e.baseline.text.length && /\s/.test(e.baseline.text[c])) c++;
  while (c > 0 && !/\s/.test(e.baseline.text[c - 1])) c--;
  runs.forEach((r) => {
    const out = [], re = /\S+\s*/g, tail = r.text.slice(c); let m;
    const sc = r.split ? r.split.char + /^\s*/.exec(r.text.slice(r.split.char))[0].length : -1;
    while ((m = re.exec(tail))) { const start = c + m.index, core = m[0].trimEnd(); out.push({ w: core, sp: m[0].slice(core.length), at: start, split: sc >= start && sc < start + core.length }); }
    r.words = out; r.score = r.rows.reduce((a, x) => a + x.dose, 0);
    if (r.dial) {
      // Each word takes the value of the last token that overlaps it (token texts join to the run's text).
      let pos = 0; const spans = r.dial.toks.map((t) => { const a = pos; pos += t.text.length; return [a, pos, t.value]; });
      for (const w of out) { let v = null; for (const [a, b, val] of spans) if (a < w.at + w.w.length && b > w.at && val != null) v = val; w.v = v; }
    }
  });
  return { e, runs, rootChar: c, shared: e.baseline.text.slice(0, c).trim() };
}
const matching = (T) => T && T.runs.find((r) => rowsKey(r.rows) === rowsKey(S.rows));
let expanded = new Set();
function runHtml(r, current, inspect) {
  if (r.live) {
    // A column of a running filmstrip: real streamed text, not clickable until it completes.
    const text = r.words.map((x) => x.w + x.sp).join("");
    return `<div class="run generating" data-n="${r.n}" aria-current="false"><div class="k"><span class="pick">running: ${esc(editsText(r.rows))}</span><span class="state info">RUNNING</span></div><div class="t">${text ? esc(text) : Array.from({ length: 5 }, (_, i) => `<span class="redact" style="width:${2.4 + (i * 3 % 5) * .7}em"></span>`).join("")}</div></div>`;
  }
  const full = expanded.has(r.n), n = full ? Infinity : current ? 26 : 16;
  const body = r.words ? r.words.slice(0, n).map((x, j) => `<span class="w${x.split ? " split" : ""}${inspect && inspect.run === r.n && inspect.i === j ? " inspected" : ""}" data-run="${r.n}" data-i="${j}"${x.v != null ? ` data-v="${x.v}"` : ""} tabindex="0" role="button" aria-label="Inspect “${esc(x.w)}”">${esc(x.w)}</span>${x.sp}`).join("") + (r.words.length > n ? `… <button class="more" type="button" data-more="${r.n}">show all</button>` : "")
    : Array.from({ length: 7 }, (_, i) => `<span class="redact" style="width:${2.4 + (i * 3 % 5) * .7}em"></span>`).join("") + `<span class="meta"> ${S.running ? "generating" : "not run yet"}</span>`;
  const edits = editsText(r.rows) + (r.research ? ", research mode" : "");
  const label = r.n ? `run ${r.n}: ${edits}` : `next: ${edits}`;
  const tag = !r.n ? `<span class="state ${S.running ? "info" : "warn"}">${S.running ? "RUNNING" : "NOT RUN"}</span>` : r.n > 1 && !r.split ? '<span class="state warn">NO SPLIT</span>' : "";
  const dl = r.dial && r.words ? `<span class="meta" style="color:var(--pen)">reading ${esc(r.dial.trait)}${r.words.some((x) => x.v >= 1) ? ", crosses 1.0" : ""}</span>` : "";
  return `<div class="run${!r.n && S.running ? " generating" : ""}${r.dial ? " dialed" : ""}" data-n="${r.n}" aria-current="${current}"><div class="k"><button class="pick" type="button" data-pick="${r.n}" ${!r.n || S.running ? "disabled" : ""}>${label}</button>${current ? '<span class="state info">SELECTED</span>' : ""}${tag}${dl}</div><div class="t">${body}</div></div>`;
}
let drawN = null;
function renderTree() {
  const T = treeRuns(), m = matching(T);
  S.sel = m ? m.n : 0;
  const list = T ? [...T.runs] : [];
  if (S.live) S.live.cols.forEach((c, i) => {
    if (c.dose === 0 && T) return;
    const text = c.pieces.join(""), from = T ? Math.min(T.rootChar, text.length) : 0, out = [], re = /\S+\s*/g; let mm;
    while ((mm = re.exec(text.slice(from)))) { const core = mm[0].trimEnd(); out.push({ w: core, sp: mm[0].slice(core.length) }); }
    list.push({ n: -(i + 1), live: true, rows: c.dose === 0 ? [] : [{ trait: S.live.trait, dose: c.dose }], words: out, score: c.dose });
  });
  else if (!m && (T || S.running)) list.push({ n: 0, rows: S.rows, words: null, score: -1e9 });
  if (!T && !S.running) {
    $("root").innerHTML = ""; $("runs").innerHTML = `<p class="empty tree-empty">${S.api ? "No runs for this prompt yet. Set the edits in the sentence, then press Run. The baseline and your edited answer appear here, and every later run adds a branch." : "The lobot server is not reachable. Start it with: uv run python server.py"}</p>`;
    $("links").innerHTML = ""; $("shareBtn").disabled = true; $("plan").textContent = ""; return;
  }
  const sel = m && m.split ? m : null;
  $("root").innerHTML = `${esc(T ? T.shared : $("prompt").value ? "…" : "")}<span class="flag" id="splitFlag">split · ${sel ? at(sel.split) : ""}</span>`;
  $("splitFlag").hidden = !sel;
  // Highest total dose on top, the no-edit run in its place, the next setup last.
  $("runs").innerHTML = list.sort((a, b) => b.score - a.score).map((r) => runHtml(r, r.n === S.sel, S.inspect)).join("");
  $("plan").textContent = m ? `matches run ${m.n}` : T ? "changed since the last run" : "";
  $("shareBtn").disabled = !(T && T.runs.some((r) => r.split));
  if (!S.running) {
    if (sel) setStatus("ok", "SPLIT", `Run ${sel.n} leaves run 1 at ${at(sel.split)}. ${sel.seconds} s, greedy, seed 0.${sel.research ? " Research mode: the guard was off." : ""}${sel.proof ? " " + proofText(sel.proof) : ""}`);
    else if (m && m.n > 1) setStatus("warn", "NO SPLIT", `Run ${m.n} matches run 1 token for token. Try a larger dose.`);
    else if (m) setStatus("info", "NO EDIT", "Run 1 is the model with no edit.");
    else setStatus("info", "READY", T ? "This setup has no run yet. Press Run." : "Press Run.");
  }
  layout();
}
function layout() {
  if ($("vWork").hidden) return;
  const tree = $("tree"), box = tree.getBoundingClientRect(), rb = $("root").getBoundingClientRect();
  const kids = [...$("runs").querySelectorAll(".run")]; if (!kids.length) { $("links").innerHTML = ""; return; }
  const stacked = matchMedia("(max-width: 860px)").matches, T = treeRuns();
  const x0 = stacked ? rb.left - box.left + 12 : rb.right - box.left + 8, y0 = stacked ? rb.bottom - box.top + 4 : rb.top - box.top + rb.height / 2;
  const cs = getComputedStyle(root);
  $("links").innerHTML = kids.map((el) => {
    const n = +el.dataset.n, b = el.getBoundingClientRect(), x1 = b.left - box.left, y1 = b.top - box.top + 12, cur = n === S.sel;
    const r = T && T.runs.find((x) => x.n === n), hot = cur && !!(r && r.split);
    const c = hot ? cs.getPropertyValue("--cut") : cur || n === 1 ? cs.getPropertyValue("--ink") : cs.getPropertyValue("--ink-muted");
    const d = stacked ? `M${x0} ${y0} C ${x0} ${y1}, ${x0} ${y1}, ${x1} ${y1}` : `M${x0} ${y0} C ${x0 + (x1 - x0) * .55} ${y0}, ${x0 + (x1 - x0) * .45} ${y1}, ${x1} ${y1}`;
    return `<path class="${n === drawN ? "draw" : ""}" d="${d}" fill="none" stroke="${c}" stroke-width="${hot ? 4 : cur ? 3 : 2}"${n ? "" : ' stroke-dasharray="5 6"'}/>`;
  }).join("") + kids.map((el) => strip(el, box, +el.dataset.n === drawN)).join("");
  drawN = null;
}
// Channel strip (read dial) under a run's words: band 0..1, one pen step per word, an ink tick where it first crosses 1.0.
new ResizeObserver(() => requestAnimationFrame(layout)).observe($("tree"));
$("runs").addEventListener("click", (e) => {
  const more = e.target.closest("[data-more]"); if (more) { expanded.add(+more.dataset.more); renderTree(); return; }
  const p = e.target.closest("[data-pick]"); if (p && !S.running) { const T = treeRuns(), r = T && T.runs.find((x) => x.n === +p.dataset.pick); if (r) { S.rows = r.rows.map((x) => ({ ...x })); S.focusEdit = 0; update(); } return; }
  const w = e.target.closest(".w"); if (w) inspect(+w.dataset.run, +w.dataset.i);
});
$("runs").addEventListener("keydown", (e) => { const w = e.target.closest(".w"); if (w && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); inspect(+w.dataset.run, +w.dataset.i); } });

/* ---------- word inspector (design decision 14A) and the reader (U2): neurons at a word, neuron card, find ---------- */
const NEUR = {}, CARDS = {}; // caches: neurons per (experiment, run, word); cards per model and neuron
const readerWords = store.get("readerWords", {}); // neuron id -> words the reader predicted it pushes toward (for the lift proof)
const plainTok = (t) => String(t).replace(/^[\sĠ▁]+/, "");
// Tokens like " storm", "storm" and " Storm" read as one word on screen: keep the first of each.
const uniq = (ws) => { const seen = new Set(); return ws.filter((w) => w && !seen.has(w.toLowerCase()) && seen.add(w.toLowerCase())); };
function inspect(n, i) {
  S.inspect = { run: n, i }; S.card = null; openInspector(); renderTree();
  loadNeurons(n, i); $("iClose")?.focus({ preventScroll: true });
}
function openInspector() { $("work").classList.add("inspecting"); $("inspector").hidden = false; renderInspector(); }
function pushBtn(L, N, words) {
  const id = `L${L}N${N}`, inEdits = S.rows.some((r) => !isB(r) && nid(r) === id);
  const why = inEdits ? "" : S.rows.some(isB) ? "remove the behavior first" : S.rows.length >= S.limits.max_rows ? "4 edits at most" : "";
  return inEdits ? `<span class="meta">in edits</span>` : `<button class="btn btn-sm" type="button" data-push="${L},${N}" data-words="${esc((words || []).join("\u0001"))}" ${why || S.running ? "disabled" : ""} ${why ? `title="${why}"` : ""}>Push</button>`;
}
function renderInspector() {
  const el = $("inspector"); if (el.hidden) return;
  const head = (h, sub) => `<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px"><h3 id="iHead">${h}</h3><button class="btn btn-sm" type="button" id="iClose">Close · Esc</button></div>${sub ? `<p class="meta" style="margin:6px 0 0">${sub}</p>` : ""}`;
  if (S.card) {
    const { layer: L, neuron: N } = S.card, c = CARDS[`${S.model}|${L}|${N}`], id = `L${L}N${N}`;
    let body;
    if (!c) body = `<p class="meta">Loading the neuron card.</p>`;
    else if (c.error) body = `<p class="empty">${esc(c.error)}</p>`;
    else {
      const pt = c.pushes_toward || {}, up = uniq((pt.up || []).map((x) => plainTok(x.token))), down = uniq((pt.down || []).map((x) => plainTok(x.token)));
      const readable = pt.coherence != null && pt.coherence >= 0.5;
      const fires = c.fires_on ? c.fires_on.slice(0, 3).map((f) => { const [a, b] = f.mark || [0, 0], t = f.context || "";
        return `<li style="margin:6px 0">${esc(t.slice(0, a))}<u style="text-decoration-thickness:2px;text-underline-offset:4px;font-weight:700">${esc(t.slice(a, b))}</u>${esc(t.slice(b))} <span class="meta">${num(f.activation)}</span></li>`; }).join("")
        : null;
      body = `<span class="label" style="margin-top:16px">Pushes toward</span>
        ${readable ? `<p style="margin:0">${esc(up.slice(0, 8).join(", "))}</p>${down.length ? `<details><summary class="meta">Pushes away from</summary><p style="margin:4px 0 0">${esc(down.slice(0, 8).join(", "))}</p></details>` : ""}`
          : `<p style="margin:0">Not readable at this layer.</p>`}
        <p class="meta" style="margin:4px 0 0">coherence <span class="mono">${pt.coherence != null ? pt.coherence.toFixed(2) : "n/a"}</span> · layer ${Math.round((pt.layer_frac ?? 0) * 100)}% deep</p>
        <span class="label" style="margin-top:16px">Fires on</span>
        ${fires ? `<ol style="margin:0;padding-left:20px">${fires}</ol>` : `<p class="meta" style="margin:0">No fires-on cache for this model yet. The model slot builds it (R3).</p>`}
        <div class="row2">${pushBtn(L, N, readable ? up.slice(0, 3) : [])}${S.inspect ? `<button class="btn btn-sm" type="button" id="iBack">Back to the word</button>` : ""}</div>
        <p class="meta" style="margin-top:12px">Neurons often mix several jobs. Run tests what one actually does.</p>`;
    }
    el.innerHTML = head(`<span class="mono" style="font-size:24px">${id}</span>`, "neuron card") + body;
    return;
  }
  if (!S.inspect) { el.hidden = true; $("work").classList.remove("inspecting"); return; }
  const T = treeRuns(), r = T && T.runs.find((x) => x.n === S.inspect.run), x = r && r.words[S.inspect.i];
  if (!x) { closeInspector(); return; }
  const k = nKey(r, S.inspect.i), d = NEUR[k];
  let list;
  if (!d) list = `<p class="meta" style="margin:0">Measuring the neurons at this word. The model reads the answer once.</p>`;
  else if (d.error) list = `<p class="meta" style="margin:0">${esc(d.error)}</p>`;
  else if (d.blocked) list = `<p class="meta" style="margin:0">The refusal guard blocked these edits, so nothing was measured.</p>`;
  else list = `<ol class="nlist" style="list-style:none;margin:0;padding:0">${d.neurons.map((nn) => `<li style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;padding:8px 0;border-top:1px solid var(--grid)">
      <div style="min-width:0"><button class="link mono" type="button" data-card="${nn.layer},${nn.neuron}">L${nn.layer}N${nn.neuron}</button> <span class="meta">${num(nn.activation)}</span>
      ${nn.pushes_toward_top ? `<br><span class="meta">pushes toward ${esc(uniq(nn.pushes_toward_top.map(plainTok)).join(", "))}</span>` : ""}</div>
      ${pushBtn(nn.layer, nn.neuron, uniq((nn.pushes_toward_top || []).map(plainTok)))}</li>`).join("")}</ol>`;
  el.innerHTML = head(`“${esc(x.w)}”`, `run ${r.n}, word ${S.inspect.i + 1} after “${esc(T.shared)}”${x.split ? ", the split" : ""}`) + `
    <div class="kv"><span class="meta">Applied edits in this run</span><b class="mono">${r.rows.length ? r.rows.map((y) => isB(y) ? esc(`${y.trait} ${fd(y.dose)}`) : `<button class="link mono" type="button" data-card="${y.layer},${y.neuron}" aria-label="Open the card for ${nid(y)}">${nid(y)}</button> ${fd(y.dose)}`).join(", ") : "none"}</b>
      <span class="meta">Observed projection</span><b class="mono">${x.v != null ? `${esc(r.dial.trait)} ${x.v.toFixed(2)}` : "n/a"}</b></div>${x.v != null ? `<p class="meta" style="margin:4px 0 0">Observed projection: 0 = neutral mean, 1 = ${esc(r.dial.trait)} mean.</p>` : ""}
    <span class="label" style="margin-top:16px">Top neurons at this word</span>${list}
    <p class="meta" style="margin-top:12px">Ranked by activation with this run's edits on. Open a neuron to see what it fires on and pushes toward.</p>`;
}
const nKey = (r, i) => `${S.model}|${$("prompt").value.trim()}|${r.n}|${rowsKey(r.rows)}|${i}`;
async function loadNeurons(n, i) {
  const T = treeRuns(), r = T.runs.find((x) => x.n === n), x = r.words[i], k = nKey(r, i);
  if (NEUR[k] && !NEUR[k].error) return renderInspector();
  if (!S.api) { NEUR[k] = { error: "The lobot server is not reachable." }; return renderInspector(); }
  // Replay mode loads no weights: say so, and point at the recorded cards of this run's neurons.
  if (S.state?.replay) { NEUR[k] = { error: "Replay mode loads no weights, so the neurons at a word are not measured. Open a neuron of this run above to see its recorded card." }; return renderInspector(); }
  NEUR[k] = null; S.reading = (S.reading || 0) + 1; renderInspector(); renderGuard();
  const beh = r.rows.find(isB);
  try {
    const res = await postJSON("api/neurons", { prompt: T.e.prompt, rows: r.rows.filter((y) => !isB(y)), trait: beh ? beh.trait : null, dose: beh ? beh.dose : null,
      reply: r.text, reply_char: x.at, top_k: 8, guard: false, research_mode: S.research });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(res.status === 409 ? "A run is in progress. Try again when it ends." : typeof body.detail === "string" ? body.detail : `The server returned HTTP ${res.status}.`);
    NEUR[k] = body;
  } catch (e) { NEUR[k] = { error: String(e.message || e) }; }
  S.reading--; renderGuard();
  if (S.inspect && S.inspect.run === n && S.inspect.i === i) renderInspector();
}
async function openCard(L, N) {
  S.card = { layer: L, neuron: N }; openInspector(); $("iClose")?.focus({ preventScroll: true });
  const k = `${S.model}|${L}|${N}`; if (CARDS[k] && !CARDS[k].error) return;
  CARDS[k] = null;
  try {
    const res = await fetch(`api/neuron/${L}/${N}?k=12`), body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `The server returned HTTP ${res.status}.`);
    CARDS[k] = body;
  } catch (e) { CARDS[k] = { error: String(e.message || e) }; }
  if (S.card && S.card.layer === L && S.card.neuron === N) renderInspector();
}
function closeInspector() { S.inspect = null; S.card = null; $("inspector").hidden = true; $("work").classList.remove("inspecting"); renderTree(); }
$("inspector").addEventListener("click", (e) => {
  const t = e.target;
  if (t.closest("#iClose")) closeInspector();
  else if (t.closest("#iBack")) { S.card = null; renderInspector(); }
  else if (t.closest("[data-card]")) { const [L, N] = t.closest("[data-card]").dataset.card.split(",").map(Number); openCard(L, N); }
  else if (t.closest("[data-push]")) {
    const b = t.closest("[data-push]"), [L, N] = b.dataset.push.split(",").map(Number), words = b.dataset.words ? b.dataset.words.split("\u0001") : [];
    S.rows.push({ layer: L, neuron: N, dose: 1 }); S.focusEdit = S.rows.length - 1;
    if (words.length) { readerWords[`L${L}N${N}`] = words; store.set("readerWords", readerWords); }
    update(); renderInspector(); setStatus("info", "READY", `L${L}N${N} is in the sentence at +1.00. Press Run to see what it does.`);
  }
});

/* Find a neuron (shelf): neurons that push one word hardest. */
async function findNeurons(word) {
  const out = $("findOut"); word = word.trim(); if (!word) return;
  out.innerHTML = `<p class="meta">Searching every layer.</p>`;
  try {
    const res = await fetch(`api/find?word=${encodeURIComponent(word)}&k=8`), body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `The server returned HTTP ${res.status}.`);
    out.innerHTML = body.results.length ? body.results.map((x) => `<button class="exp" type="button" data-card="${x.layer},${x.neuron}"><span class="p mono">L${x.layer}N${x.neuron}</span><span class="meta">${x.rank ? `“${esc(word)}” is its #${x.rank} word` : "pushes it, not in its top 100"}</span></button>`).join("")
      : `<p class="empty">No neuron pushes “${esc(word)}” clearly.</p>`;
  } catch (e) { out.innerHTML = `<p class="empty">${esc(e.message || e)}</p>`; }
}
$("findForm").addEventListener("submit", (e) => { e.preventDefault(); findNeurons($("findIn").value); });
$("findOut").addEventListener("click", (e) => { const b = e.target.closest("[data-card]"); if (!b) return; const [L, N] = b.dataset.card.split(",").map(Number); setView("work"); openCard(L, N); });

/* The lift proof (U2): after a run with a neuron the reader described, measure how much it lifted the predicted word. */
async function proveLift(n) {
  const T = treeRuns(), r = T && T.runs.find((x) => x.n === n); if (!r || !S.api || S.state?.replay) return;
  // Words the reader predicted: from the Push that added the neuron, else from its card (only when readable).
  let row = r.rows.find((y) => !isB(y) && readerWords[nid(y)]);
  if (!row) {
    for (const y of r.rows.filter((z) => !isB(z))) {
      const k = `${S.model}|${y.layer}|${y.neuron}`;
      if (!CARDS[k]) { try { const res = await fetch(`api/neuron/${y.layer}/${y.neuron}?k=12`); if (res.ok) CARDS[k] = await res.json(); } catch { /* no card, no proof */ } }
      const pt = CARDS[k]?.pushes_toward;
      if (pt && pt.coherence >= 0.5) { const w = uniq((pt.up || []).map((x) => plainTok(x.token))).slice(0, 3); if (w.length) { readerWords[nid(y)] = w; row = y; break; } }
    }
    if (!row) return;
  }
  const words = readerWords[nid(row)].slice(0, 3);
  try {
    const res = await postJSON("api/lift", { layer: row.layer, neuron: row.neuron, dose: row.dose, words, prompts: null }), b = await res.json();
    if (!res.ok || !b.base_p || !(b.base_p[0] > 0)) return;
    r.proof = { word: b.words[0], ratio: b.steered_p[0] / b.base_p[0], id: nid(row), dose: row.dose };
    const e = curExp(), run = e && e.runs.find((x) => x.n === n); if (run) { run.proof = r.proof; saveExps(); }
    if (S.sel === n) renderTree();
  } catch { /* no proof line without a real measurement */ }
}
const proofText = (p) => `The reader predicted “${p.word}” for ${p.id}; pushing it at ${fd(p.dose)} made “${p.word}” ${p.ratio >= 10 ? Math.round(p.ratio) : p.ratio.toFixed(1)}× more likely as the next word (8 neutral prompts, measured).`;



async function refreshState() {
  try { const r = await fetch("api/state"); if (r.ok) { applyState(await r.json()); update(); } } catch { /* keep last state */ }
}
function applyState(st) {
  S.state = st; S.model = st.active_model;
  S.limits = { layers: st.layers, neurons: st.neurons, dose_min: st.dose_min, dose_max: st.dose_max, max_rows: st.max_rows };
  K.draw();
}
function update() { renderShelf(); renderSentence(); renderTree(); renderKnob(); renderGuard(); renderInspector(); }
// Seed the browser's experiments with every recording (example.json, sweep*.json) once per file version,
// so the tree is not empty on first visit and a newly recorded sweep shows up.
async function seedRecorded() {
  const seen = new Set(store.get("seededFiles", []));
  const names = ["example.json", "sweep.json", ...(S.state ? S.state.models.map((m) => `sweep.${m.short}.json`) : [])];
  for (const name of names) {
    const text = await fetch(name).then((r) => (r.ok ? r.text() : null)).catch(() => null);
    const tag = text && `${name}:${text.length}`;
    if (!text || seen.has(tag)) continue;
    const d = JSON.parse(text);
    if (d.runs) for (const r of d.runs) addResult({ model_id: d.model_id, prompt: d.prompt, rows: r.rows, baseline: d.baseline, steered: { text: r.text, tokens: r.tokens }, split: r.split, seconds: r.seconds, recipe: r.recipe || "", post: r.post || "", dial: r.dial });
    else addResult({ ...d });
    seen.add(tag);
  }
  store.set("seededFiles", [...seen]);
}

export { renderShelf, STAGES, NST, active, behaviors, renderBehaviors, useBehavior, renderSentence, readable, readMenu, addMenu, pop, closePop, openPop, rowForm, treeRuns, matching, expanded, runHtml, drawN, renderTree, layout, NEUR, readerWords, plainTok, uniq, inspect, openInspector, pushBtn, renderInspector, nKey, loadNeurons, openCard, closeInspector, findNeurons, proveLift, proofText, refreshState, applyState, update, seedRecorded };
