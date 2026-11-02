import { update } from "./workbench.js";
import { KEY_HELP, RULES, SCREEN, SHELF_KEYS, VERDICT, ci, compareSection, doseRuleLine, exOut, exState, explainSection, explains, explorerSection, followupCtl, guardSection, markSplit, mins, noWeights, num, reports, setExDose, short, splitWord, stroke, tuningSection, vecPoint, wireExplorer } from "./evidence.js";
import { libraryPage } from "./library.js";
import { legend } from "../components/legend.js";
import { blockedMsg, confirmDlg, creating, renderGuard, run, stop, stream } from "./run.js";
import { setView } from "./share.js";
import { NST, STAGES, active, behaviors, closeInspector, closePop, inspect, pop, renderShelf, useBehavior } from "./workbench.js";
import { strip } from "../components/strip.js";
import { $, S, at, eid, esc, fd, isB, rowsKey } from "../state.js";
import { getJSON, postJSON } from "../api.js";
/* ---------- 100X on screen: describe in words, the agent, self-labeling, chat, and evidence actions ---------- */
const errText = async (r) => { const b = await r.json().catch(() => ({})); return typeof b.detail === "string" ? b.detail : b.detail?.reason || `The server returned HTTP ${r.status}.`; };
const DR = {}; // drafts: id -> { draft, suggest, result, err, busy }
const AGENT_KEY = { running: ["info", "RUNNING"], done: ["ok", "DONE"], stopped: ["warn", "STOPPED"], out_of_rounds: ["warn", "OUT OF ROUNDS"], refused: ["err", "REFUSED"], failed: ["err", "FAILED"] };
const CH = { sid: null, key: null, turns: [], busy: false, err: "" };
const chatKey = () => `${S.model}|${rowsKey(S.rows)}|${S.read || ""}|${S.research}`;
const ACT = {}; // per job: { verify, del, targets, install, iverify }
function actionsSection(R, id, b) {
  const a = ACT[id] || (ACT[id] = {}), off = noWeights(), v = a.verify, d = a.del, name = b?.name;
  const vOut = !v ? "" : v.busy ? `<p class="meta">Re-checking: <span class="mono">${v.done ?? 0}</span> of <span class="mono">${v.total ?? "?"}</span> prompts.</p>` : v.err ? `<p class="meta" style="color:var(--error)">${esc(v.err)}</p>`
    : `<p><span class="state ${v.res.pass ? "ok" : "warn"}">${v.res.pass ? "HOLDS" : "DOES NOT HOLD"}</span> Measured ${num(v.res.measured_mean)} on <span class="mono">${v.res.n}</span> held-out prompts; the patch shipped ${num(v.res.shipped_mean)} ${ci(v.res.shipped_ci)}. ${v.res.pass ? "The measured mean is inside the shipped interval." : "The measured mean is outside the shipped interval."}</p>`;
  const col = (c, base, cls) => `<div class="br"><div class="bl mono">${c.mode === "on" ? "behavior removed" : "as is"}</div><p class="bt">${base != null && c.mode === "on" ? markSplit(c.text, base, cls) : esc(c.text)}</p></div>`;
  const dOut = !d ? "" : d.busy ? `<p class="meta">Running. The guard first checks the removal with full generation, which can take a few minutes; then both answers stream.</p>` : d.err ? `<p class="meta" style="color:var(--error)">${esc(d.err)}</p>`
    : `<div class="fork">${col(d.cols[0], null)}${col(d.cols[1], d.cols[0].text, "ink")}</div><p class="meta">Same prompt, same seed. The direction is removed at layer ${d.layer ?? "?"}; the first changed word is underlined.</p>
      ${splitWord(d.cols[0].text, d.cols[1].text) == null ? `<p class="note"><span class="state info">NO CHANGE</span> The two answers are the same. The behavior was not in this answer, so removing it changed nothing. Try a prompt that asks for the behavior, for example “Speak like ${esc(name || "it")}. Tell me about your favorite food.”</p>` : ""}`;
  const t = a.targets, ins = a.install, iv = a.iverify;
  const tOut = !t ? "" : t.err ? `<p class="meta" style="color:var(--error)">${esc(t.err)}</p>` : `<ol class="ruled screens">${t.list.map((x) => `<li><div><b>${x.target === "llamacpp" ? "llama.cpp" : "Ollama"}</b><br><span class="meta">${esc(x.version || x.detail || "")}</span>
      ${x.available ? `<div class="row2"><label style="flex:1;min-width:220px"><span class="meta">${x.target === "llamacpp" ? "Path to the model GGUF" : "Ollama base model"}</span><input class="num" type="text" spellcheck="false" id="mref-${x.target}" style="font-family:var(--mono)" placeholder="${x.target === "llamacpp" ? "models/llama-3.2-3b-instruct-f16.gguf" : "llama3.2:3b"}"></label><button class="btn btn-sm" type="button" data-install="${x.target}" data-id="${esc(id)}" ${ins?.busy ? "disabled" : ""}>Install</button></div>` : ""}</div>
      <span class="state ${x.available ? "ok" : "info"}">${x.available ? "FOUND" : "NOT FOUND"}</span></li>`).join("")}</ol>`;
  const iOut = !ins ? "" : ins.busy ? `<p class="meta">Installing.</p>` : ins.err ? `<p class="meta" style="color:var(--error)">${esc(ins.err)}</p>`
    : `<p><span class="state ok">INSTALLED</span> ${esc(ins.res.detail || "")}</p>${ins.res.run ? `<div class="copyrow"><div><span class="label">Run it</span><p class="meta v" id="irun-${esc(id)}" style="margin:0">${esc(runLine(ins.res.run))}</p></div><button class="btn btn-sm" type="button" data-copy-el="irun-${esc(id)}">Copy</button></div>` : ""}
      <div class="row2"><button class="btn btn-sm" type="button" data-iverify="${esc(ins.res.install_id)}" data-id="${esc(id)}" ${iv?.busy ? "disabled" : ""}>Check the installed patch</button><span class="meta">Runs 16 of the patch's own held-out prompts in ${ins.res.target === "llamacpp" ? "llama.cpp" : "Ollama"}, with and without it, and judges them here.</span></div>
      ${!iv ? "" : iv.busy ? `<p class="meta">Checking the installed patch.</p>` : iv.err ? `<p class="meta" style="color:var(--error)">${esc(iv.err)}</p>` : `<p><span class="state ${iv.res.comparison?.pass ? "ok" : "warn"}">${iv.res.comparison?.pass ? "HOLDS" : "DOES NOT HOLD"}</span> measured ${num(iv.res.comparison?.measured_mean)} on ${iv.res.comparison?.n} prompts; shipped ${ci(iv.res.comparison?.shipped_ci)}.</p>`}`;
  return `<h2 class="sec">Use it elsewhere</h2>
    <h3 class="sub">Re-check on this machine</h3><p class="meta">Re-runs 16 of the patch's held-out prompts on the loaded model and compares the effect with the interval the patch shipped with.</p>
    <div class="row2"><button class="btn" type="button" data-verify="${esc(id)}" ${off || v?.busy || !b?.patch ? "disabled" : ""}>Re-check</button>${off ? `<span class="meta">${off}</span>` : !b?.patch ? `<span class="meta">Only a patch can be re-checked.</span>` : ""}</div>${vOut}
    ${b?.del && !b.newest ? `<h3 class="sub">Remove it from one answer</h3><p class="meta">Removing runs on the newest version of ${esc(b.name)}. Open the newest version to try it.</p>` : ""}
    ${b?.del && b.newest ? `<h3 class="sub">Remove it from one answer</h3><p class="meta">Removes the behavior's direction from the model on one prompt, next to the answer as is.</p>
    <label class="label" for="delIn-${esc(id)}">Prompt</label><textarea class="field" rows="1" id="delIn-${esc(id)}">${esc(a.delPrompt || "Tell me about the ocean.")}</textarea>
    <div class="row2"><button class="btn" type="button" data-del="${esc(id)}" ${off || d?.busy ? "disabled" : ""}>Run with it removed</button>${off ? `<span class="meta">${off}</span>` : ""}</div>${dOut}` : ""}
    <h3 class="sub">Install into llama.cpp or Ollama</h3><p class="meta">Exports the patch as a control vector for the tool you run models with. Nothing is sent anywhere.</p>
    <div class="row2"><button class="btn" type="button" data-targets="${esc(id)}" ${!b?.patch ? "disabled" : ""}>Find install targets</button></div>${tOut}${iOut}`;
}
// The install's run line: llama.cpp gives {base_cmd, steer_flags}; Ollama gives a string or a list.
const quoteArg = (x) => (/[\s"'$]/.test(x) ? `'${String(x).replace(/'/g, "'\\''")}'` : String(x));
function runLine(run) {
  if (!run) return "";
  if (typeof run === "string") return run;
  if (Array.isArray(run)) return run.map(quoteArg).join(" ");
  if (run.base_cmd) return [...run.base_cmd, ...(run.steer_flags || []), "-p", "Tell me about the ocean."].map(quoteArg).join(" ");
  return JSON.stringify(run);
}
async function actVerify(id) {
  const a = ACT[id]; a.verify = { busy: true }; renderPage();
  try { await stream("api/verify", { trait_id: id, quick: true }, (ev) => {
      if (ev.stage === "progress") { a.verify.done = ev.done; a.verify.total = ev.total; renderPage(); }
      else if (ev.stage === "done") { a.verify = ev.result && ev.result.measured_mean != null ? { res: ev.result } : { err: ev.result?.error || "The check returned no result." }; return true; }
      return false; });
  } catch (e) { a.verify = { err: e.message }; }
  if (a.verify.busy) a.verify = { err: "The check stopped before it finished." };
  renderPage();
}
async function actDelete(id, name) {
  const a = ACT[id]; a.delPrompt = $(`delIn-${id}`)?.value.trim() || "Tell me about the ocean."; a.del = { busy: true }; renderPage();
  const cols = [{ mode: "off", text: "" }, { mode: "on", text: "" }];
  try { await stream("api/delete", { prompt: a.delPrompt, trait: name, guard: !S.research, research_mode: S.research, max_new_tokens: 120 }, (ev) => {
      if (ev.stage === "token") cols[ev.col].text += ev.text;
      else if (ev.stage === "blocked") throw new Error(blockedMsg(ev.guard, `removing ${name}`).replace(" Lower the dose, or use Research mode.", " Removing a direction can also remove part of the model's refusals. Research mode runs it anyway and records that."));
      // The removal stream ends with "complete" (the other streams end with "done").
      else if (ev.stage === "complete" || ev.stage === "done") { const r = ev.result || {}; a.del = { cols: r.columns || cols, layer: r.layer }; return true; }
      else if (ev.stage === "stopped") throw new Error(`The removal stopped (after: ${ev.after}).`);
      return false; });
  } catch (e) { a.del = { err: e.message }; }
  if (a.del.busy) a.del = { err: "The run stopped before it finished." };
  renderPage();
}
// Every version of this behavior on this model, newest first; the open one is marked.
function versionsSection(name, id) {
  const vs = behaviors().all.filter((x) => x.name === name).sort((a, b) => b.version - a.version);
  if (vs.length < 2) return "";
  const rows = vs.map((v) => { const e = v.ev || {}, vk = VERDICT[e.pareto_verdict];
    return `<li${v.id === id ? ' aria-current="true"' : ""}><div><button class="link" type="button" data-beh="${esc(v.id)}"><b>v${v.version}</b></button> <span class="meta">${v.dose != null ? `layer ${v.layer} · dose ${fd(v.dose)}` : "no dose"}${e.fluency != null ? ` · fluency ${e.fluency.toFixed(2)}` : ""}${e.dose_rule ? " · fluency-aware dose" : ""}${v.id === id ? " · open now" : ""}</span></div>
      <span>${vk ? `<span class="state ${vk[0]}">${vk[1]}</span> ` : ""}<span class="state ${v.kind}">${v.key}</span></span></li>`; }).join("");
  return `<h2 class="sec">Versions</h2><ol class="ruled screens">${rows}</ol>`;
}
function repairSection(R, id) {
  const screens = R.guard?.screens || (R.guard ? { refusal: R.guard } : {}), flagged = Object.entries(screens).filter(([, v]) => v.blocked || v.would_block).map(([n]) => n);
  const recs = R.repair || {};
  if (!flagged.length && !Object.keys(recs).length) return "";
  const head = `<h2 class="sec">Repair</h2><p class="lead">Repair removes the part of the patch direction that makes safe replies less likely, measured by gradients on prompts outside the guard set. It tries the patch dose, then lower doses, in a fixed order, and keeps the first candidate that is clean and still moves the behavior. Then it checks that candidate once with the full guard and the held-out report card.</p>`;
  const tables = Object.entries(recs).map(([m, r]) => {
    const sc = r.flagged || [];
    const rows = (r.candidates || []).map((c) => `<tr><td>${esc(c.method)}</td><td class="mono">${c.factor.toFixed(2)}×</td><td class="mono">${fd(c.dose)}</td>${sc.map((s) => `<td class="mono">${c.screens?.[s] ? (c.screens[s].blocked_prompts?.length ?? "?") : "–"}</td>`).join("")}<td class="mono">${c.tuning_gain != null ? num(c.tuning_gain) : "–"}</td><td>${c.ok ? '<span class="state ok">CLEAN</span>' : '<span class="state warn">FLAGGED</span>'}</td></tr>`).join("");
    const st = r.status === "repaired" ? ["ok", "REPAIRED"] : r.status === "clean" ? ["ok", "NOTHING TO REPAIR"] : ["warn", "NOT REPAIRED"];
    const before = r.before?.screens || {};
    return `<h3 class="sub">${m === "gradient" ? "Gradient method" : "Safety-direction method (older)"} <span class="state ${st[0]}">${st[1]}</span></h3>
      <p class="meta">Before: ${sc.map((s) => `${SCREEN[s] || s} ${before[s]?.blocked_prompts?.length ?? "?"} flagged`).join(", ") || "clean"}.${r.projection ? ` Removed <span class="mono">${(r.projection.removed_share * 100).toFixed(1)}%</span> of the direction.` : ""}</p>
      ${rows ? `<div class="evwrap"><table class="ev pt"><thead><tr><th scope="col">Method</th><th scope="col">Dose factor</th><th scope="col">Dose</th>${sc.map((s) => `<th scope="col">${SCREEN[s] || esc(s)} flagged</th>`).join("")}<th scope="col">Tuning effect</th><th scope="col">Result</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}
      ${r.status === "repaired" ? `<div class="row2"><a class="btn" href="api/traits/${encodeURIComponent(id)}/repaired/${m}" download>Download repaired patch</a><span class="meta">Its guard and card were measured once, after the choice.</span></div>` : ""}`;
  }).join("");
  return head + tables + (flagged.length ? followupCtl("repair", id, R, Object.keys(recs).length ? "Run repair again" : "Try repair", "About 10 to 20 min. It needs torch for the gradients.") : "");
}
function openPage(p) { S.page = p; closePop(); if (S.inspect || S.card) closeInspector(); setView("page"); renderPage(true); }
async function renderPage(first = false) {
  const el = $("vPage"), p = S.page; if (!p || el.hidden) return;
  const keep = document.activeElement && el.contains(document.activeElement) ? document.activeElement.id : null;
  const top = (h) => `<div class="top"><h1 id="pageH" tabindex="-1">${h}</h1><button class="btn" type="button" data-back>Back to the tree</button></div>`;
  if (p.kind === "new") {
    if (!S.families) { el.innerHTML = top("New behavior") + `<p class="meta">Loading the families.</p>`; try { S.families = (await getJSON("api/families")).families; } catch (e) { el.innerHTML = top("New behavior") + `<p class="empty">The families did not load (${esc(e.message)}).</p>`; return; } }
    const job = creating(), llama = /llama-3\.2-3b/i.test(S.model || "");
    if (!AG.list) await loadAgents();
    el.innerHTML = top("New behavior") + describeBox() + `<h2 class="sec">Or start from a validated family</h2><p class="lead">Create builds a behavior on <span class="mono">${esc(short(S.model))}</span> from a validated family: contrast prompts, a direction, a layer and dose, a held-out final check, and a report card. Only families with a hand-validated judge can pass.</p>
      <p class="meta">${llama ? "About 50 min on Llama 3.2 3B, measured on one run." : "Not measured on this model yet."} The server keeps working if you close this page. Run is off while Create works.${job ? ` One Create at a time: ${esc(job.trait)} is running.` : ""}</p>
      <ol class="ruled">${S.families.map((f) => { const ok = f.hand_validated, v = f.validation || {};
        return `<li class="${ok ? "" : "off"}"><div><b class="name">${esc(f.name)}</b>${f.aliases?.length ? ` <span class="meta">also ${esc(f.aliases.join(", "))}</span>` : ""}<br>
          <span class="meta">${ok ? `judge asks: “${esc(f.judge)}” · agreement ${v.agreement != null ? v.agreement.toFixed(2) : "n/a"} on ${v.hand_labels ?? "?"} hand labels` : "not supported yet: its judge has no hand-validated labels"}</span></div>
          <button class="btn" type="button" data-create="${esc(f.name)}" ${!ok || job ? "disabled" : ""}>Create</button></li>`; }).join("")}</ol><p class="err meta" id="newErr" role="alert" style="color:var(--error)"></p>`;
  } else if (p.kind === "create") {
    let j = S.jobs.find((x) => x.id === p.id);
    if (!j || active(j)) { try { j = await getJSON(`api/create/${encodeURIComponent(p.id)}`); const i = S.jobs.findIndex((x) => x.id === j.id); if (i >= 0) S.jobs[i] = j; else S.jobs.push(j); } catch { if (!j) { el.innerHTML = top("Create") + `<p class="empty">This Create job is not on the server.</p>`; return; } } }
    const si = STAGES.findIndex(([k]) => k === j.stage), live = active(j), doneAll = j.status === "done";
    const el2 = ((live ? Date.now() / 1000 : j.updated) - j.created);
    const head = live ? `Creating ${esc(j.trait)}` : `${esc(j.trait)}: ${doneAll ? "done" : j.status}`;
    const rows = STAGES.map(([k, label], i) => { const skip = k === "pareto" && j.pareto === false, st = skip && !(i === si && live) ? "skip" : doneAll || i < si ? "done" : i === si && live ? "now" : "wait", pr = j.progress || {};
      if (st === "skip") return `<li class="wait"><span class="no">${i + 1}</span><div><b>${label}</b><br><span class="meta">off for this job (fast schedule); run it later from the evidence page</span></div><span class="state info">SKIPPED</span></li>`;
      return `<li class="${st === "done" ? "done" : st === "wait" ? "wait" : ""}" ${st === "now" ? 'aria-current="step"' : ""}><span class="no">${i + 1}</span><div><b>${label}</b><br><span class="meta">${st === "now" ? `${pr.done ?? 0} of ${pr.total ?? "?"} units` : st === "done" ? "done" : i === si && !live ? `${j.status} here` : "waiting"}</span></div>
        <span class="state ${st === "done" ? "ok" : st === "now" ? "info" : "info"}">${st === "done" ? "DONE" : st === "now" ? "NOW" : i === si && !live ? j.status.toUpperCase() : "WAITING"}</span></li>`; }).join("");
    const verdict = !doneAll ? "" : j.verdict === "knob" ? `<p><span class="state ok">KNOB</span> A layer and dose passed the gates: layer <span class="mono">${j.frozen?.layer}</span>, dose <span class="mono">${fd(j.frozen?.dose ?? 0)}</span>.</p><div class="row2"><button class="btn" type="button" data-beh="${esc(j.id)}">Open evidence</button><button class="btn" type="button" data-use="${esc(j.id)}">Use in the sentence</button></div>`
      : j.verdict === "blocked" ? `<p><span class="state err">GUARD</span> The refusal guard blocked the frozen direction. It cannot be used.</p>` : `<p><span class="state warn">NO KNOB</span> No layer and dose passed the gates. Nothing was saved as a knob.</p><button class="btn" type="button" data-beh="${esc(j.id)}">Open evidence</button>`;
    el.innerHTML = top(head) + `<p class="meta"><span class="mono">${esc(short(j.model_id))}</span>, seed ${j.seed}, ${live ? "elapsed" : "took"} <span class="mono">${mins(Math.max(0, el2))}</span>${j.research_mode ? ", research mode" : ""}</p>
      ${(() => { const pr = j.progress || {}, f = doneAll ? 1 : Math.min(1, (Math.max(0, si) + (live && pr.total ? (pr.done || 0) / pr.total : 0)) / NST), x = Math.round(f * 600);
        return `<svg viewBox="0 0 600 8" preserveAspectRatio="none" style="width:100%;max-width:900px;height:8px;display:block;margin-top:16px" role="img" aria-label="${Math.round(f * 100)}% of Create done"><line x1="0" y1="4" x2="${x}" y2="4" stroke="var(--ink)" stroke-width="3"/><line x1="${x}" y1="4" x2="600" y2="4" stroke="var(--ink-muted)" stroke-width="2" stroke-dasharray="2 6"/></svg>`; })()}
      <ol class="ruled stages" aria-label="Create stages">${rows}</ol>
      ${live ? `<div class="row2"><button class="btn" type="button" id="stopCreate">Stop Create</button><span class="meta">${si >= 3 ? "Stopping now uses up this version's final check." : "Stopping before the final check costs nothing."}</span></div><p class="meta">The server keeps working if you close this page. The page reattaches when you open it again.</p>` : ""}
      ${j.status === "failed" ? `<p class="empty" style="color:var(--error)">Create failed: ${esc((j.error || "").split("\n").filter(Boolean).pop() || "no reason given")}</p>` : ""}${verdict}`;
  } else if (p.kind === "behavior") {
    const b = behaviors().all.find((x) => x.id === p.id), j = S.jobs.find((x) => x.id === p.id), name = b ? b.name : j ? j.trait : "Behavior";
    if (!(p.id in reports)) { el.innerHTML = top(esc(name)) + `<p class="meta">Loading the evidence.</p>`; try { reports[p.id] = await getJSON(`api/traits/${encodeURIComponent(p.id)}/report`); } catch (e) { reports[p.id] = { missing: e.status || e.message }; } if (S.page !== p) return; }
    if (!S.families) S.families = await getJSON("api/families").then((x) => x.families).catch(() => []);
    const R = reports[p.id], fam = (S.families || []).find((f) => f.name === name), fr = R.frozen || (b && { layer: b.layer, dose: b.dose }) || {};
    const actions = `<div class="row2">${b && b.usable ? `<button class="btn" type="button" data-use="${esc(b.id)}">Use in the sentence</button>` : ""}${b && b.patch ? `<a class="btn" href="${esc(b.patch)}" download>Download patch</a>` : `<button class="btn" type="button" disabled>Download patch</button><span class="meta">${b && b.verdict === "knob" ? "Patch download needs the evidence API." : "Only a passing knob has a patch."}</span>`}</div>`;
    let body;
    if (R.missing) body = `<p class="empty">The evidence needs the evidence API (E1), which this server does not have yet. Verdict from the Create job: ${esc(b ? b.key : j?.verdict || "unknown")}.</p>`;
    else {
      const F = R.final || {}, el_ = R.eligibility || {}, d = fr.dose, P = R.pareto, pt = vecPoint(P, d);
      const verb = (label, k, card, eligible, doses, side) => { const c = card || null, fails = c && !c.passed ? Object.entries(c.rules || {}).filter(([, v]) => !v).map(([r]) => RULES[r] || r) : [];
        return `<tr><td class="verb">${stroke(k === "delete" ? "delete" : "update")}${label}</td><td>${c ? `<span class="state ${c.passed ? "ok" : "warn"}">${c.passed ? "PASS" : "FAIL"}</span>${fails.length ? `<br><span class="meta">${esc(fails.join("; "))}</span>` : ""}${c.passed && !eligible ? `<br><span class="meta">not offered: read check failed</span>` : ""}` : `<span class="state info">NOT TESTED</span>`}</td>
          <td class="mono">${c ? `${num(c.trait_mean)} ${ci(c.trait_ci)}` : ""}</td><td class="mono">${c ? `under ${num(c.control_ci?.[1])}` : ""}</td><td class="meta">${side || "not measured yet"}</td><td class="mono">${doses}</td></tr>`; };
      const upSide = pt ? `fluency <span class="mono">${pt.fluency.toFixed(2)}</span> (base ${P.base.fluency.toFixed(2)}), relevance <span class="mono">${pt.relevance.toFixed(2)}</span> (base ${P.base.relevance.toFixed(2)})` : "not measured: compare with a prompt below";
      const delSide = F.delete?.nll ? `reply NLL ${num(F.delete.nll.delta)} nats` : "";
      const rg = R.read_gate;
      body = `<p class="meta">Held-out evaluation · ${esc(name)} v${b ? b.version : "?"} · layer <span class="mono">${fr.layer ?? "?"}</span> · dose <span class="mono">${d != null ? fd(d) : "?"}</span>. Effects are judge-score gains on held-out prompts, with 95% bootstrap intervals.</p>
        ${doseRuleLine(R)}
        <div class="evwrap"><table class="ev"><thead><tr><th scope="col">Verb</th><th scope="col">Result</th><th scope="col">Effect (95% CI)</th><th scope="col">Control bound</th><th scope="col">Side effects</th><th scope="col">Tested doses</th></tr></thead><tbody>
        <tr><td class="verb">${stroke("read")}Read</td><td>${rg ? `<span class="state ${rg.pass ? "ok" : "warn"}">${rg.pass ? "PASS" : "FAIL"}</span>${rg.pass ? "" : `<br><span class="meta">the reading does not track the judge (margin under 0.10)</span>`}` : `<span class="state info">NOT TESTED</span>`}</td><td class="mono">${rg ? `margin ${num(rg.agreement?.margin)}` : ""}</td><td></td><td class="meta">none (reading only)</td><td class="mono">none</td></tr>
        ${verb("Increase", "up", F.update_up?.card, el_.update_up, d != null ? fd(d) : "", upSide)}${verb("Decrease", "down", F.update_down?.card, el_.update_down, d != null ? fd(-d) : "")}${verb("Delete", "delete", F.delete?.card, el_.delete, "ablation", delSide)}
        </tbody></table></div>
        ${compareSection(R, p.id, fam)}
        ${explorerSection(R, p.id)}
        ${tuningSection(R)}
        ${guardSection(R)}
        ${repairSection(R, p.id)}
        ${versionsSection(name, p.id)}
        ${actionsSection(R, p.id, b)}
        ${explainSection(p.id)}
        <h2 class="sec">Record</h2>
        <div class="kv2"><span>Judge</span><span>${fam ? `“${esc(fam.judge)}” · agreement <span class="mono">${fam.validation?.agreement?.toFixed(2) ?? "n/a"}</span> on ${fam.validation?.hand_labels ?? "?"} hand labels` : "unknown"}</span>
          <span>Model fingerprint</span><span class="mono">${esc(!R.fingerprint ? "not recorded" : typeof R.fingerprint === "string" ? R.fingerprint : Object.entries(R.fingerprint).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join("; "))}</span>
          <span>Create time</span><span class="mono">${R.seconds ? mins(R.seconds) : "n/a"}</span></div>`;
    }
    el.innerHTML = top(esc(name) + (b ? ` <span class="meta">v${b.version}</span>` : "")) + (b ? `<p><span class="state ${b.kind}">${b.key}</span> <span class="meta"><span class="mono">${esc(short(S.model))}</span>${b.created ? `, created ${new Date(b.created * 1000).toLocaleDateString()}` : ""}</span></p><p class="keyhelp">${esc(KEY_HELP[b.key] || "")}</p>` : "") + actions + body + legend([...SHELF_KEYS, "FAIL", "NOT TESTED", "KNOB BETTER", "NO CLEAR WINNER", "PROMPT BETTER", "NOT FLUENT"]);
    wireExplorer(p.id);
  } else if (p.kind === "library") {
    el.innerHTML = top("Library") + libraryPage();
  } else if (p.kind === "agent") {
    if (!AG.runs[p.id] || AG.runs[p.id].status === "running") { try { AG.runs[p.id] = await getJSON(`api/agent/${encodeURIComponent(p.id)}`); } catch (e) { AG.runs[p.id] = AG.runs[p.id] || { status: "failed", error: e.message, steps: [] }; } if (S.page !== p) return; }
    el.innerHTML = top("Muse builds a behavior") + agentPage(p.id);
  } else if (p.kind === "draft") {
    el.innerHTML = top("Label it yourself") + draftPage(p.id);
  } else if (p.kind === "chat") {
    el.innerHTML = top("Chat") + chatPage();
    const log = $("chatLog"); if (log) log.scrollTop = log.scrollHeight;
    requestAnimationFrame(drawChatStrips);
  }
  if (first) $("pageH")?.focus(); else if (keep && $(keep)) $(keep).focus();
}
$("vPage").addEventListener("click", async (e) => {
  const t = e.target;
  if (t.closest("[data-back]")) { S.page = null; setView("work"); update(); }
  else if (t.closest("[data-use]")) useBehavior(t.closest("[data-use]").dataset.use);
  else if (t.closest("[data-beh]")) openPage({ kind: "behavior", id: t.closest("[data-beh]").dataset.beh });
  else if (t.closest("[data-create]")) {
    const b = t.closest("[data-create]"); b.disabled = true;
    try {
      const r = await postJSON("api/create", { description: b.dataset.create, guard: !S.research, research_mode: S.research }), body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${r.status}`);
      S.jobs.push(body); openPage({ kind: "create", id: body.job_id }); update();
    } catch (err) { b.disabled = false; $("newErr").textContent = `Create did not start: ${err.message}`; }
  } else if (t.closest("[data-agent-start]")) {
    const desc = $("descIn").value.trim(); S.descDraft = desc; S.agentLane = $("laneIn").checked ? "llama" : "torch";
    if (!desc) { AG.err = "Write a behavior first."; return renderPage(); }
    const btn = t.closest("[data-agent-start]"); btn.disabled = true; AG.err = "";
    try { const r = await postJSON("api/agent", { description: desc, lane: S.agentLane, max_rounds: 3 }); if (!r.ok) throw new Error(await errText(r));
      const body = await r.json(); AG.runs[body.run_id] = { status: "running", steps: [], description: desc }; AG.list = null; openPage({ kind: "agent", id: body.run_id });
    } catch (err) { AG.err = err.message; renderPage(); }
  } else if (t.closest("[data-agent]")) { openPage({ kind: "agent", id: t.closest("[data-agent]").dataset.agent });
  } else if (t.closest("[data-agent-stop]")) { t.closest("[data-agent-stop]").disabled = true; try { await postJSON(`api/agent/${encodeURIComponent(S.page.id)}/stop`, {}); } catch { /* the next poll shows it */ }
  } else if (t.closest("[data-draft-start]")) {
    const desc = $("descIn").value.trim(); S.descDraft = desc;
    if (!desc) { AG.err = "Write a behavior first."; return renderPage(); }
    const btn = t.closest("[data-draft-start]"); btn.disabled = true; btn.textContent = "Drafting…"; AG.err = "";
    try { const r = await postJSON("api/drafts", { description: desc, drafter: "local" }); if (!r.ok) throw new Error(await errText(r));
      const d = await r.json(); DR[d.id] = { draft: { ...d, replies: d.replies.map((x) => ({ ...x, label: null })) } }; openPage({ kind: "draft", id: d.id });
    } catch (err) { AG.err = err.message; renderPage(); }
  } else if (t.closest("[data-lab]")) {
    const b2 = t.closest("[data-lab]"), x = DR[S.page.id], rid = b2.dataset.lab, val = b2.dataset.v === "1";
    try { const r = await postJSON(`api/drafts/${encodeURIComponent(S.page.id)}/labels`, { labels: [{ reply_id: rid, label: val }] }); if (!r.ok) throw new Error(await errText(r));
      const rep_ = x.draft.replies.find((y) => y.reply_id === rid); if (rep_) rep_.label = val; x.err = ""; x.result = null;
    } catch (err) { x.err = err.message; }
    renderPage(); document.querySelector(`[data-lab="${rid}"][data-v="${val ? 1 : 0}"]`)?.focus();
  } else if (t.closest("[data-draft-suggest]")) {
    const x = DR[S.page.id]; x.busy = true; renderPage();
    try { const r = await postJSON(`api/drafts/${encodeURIComponent(S.page.id)}/prelabel`, {}); if (!r.ok) throw new Error(await errText(r)); x.suggest = (await r.json()).suggestions || {}; x.err = ""; } catch (err) { x.err = err.message; }
    x.busy = false; renderPage();
  } else if (t.closest("[data-draft-accept]")) {
    const x = DR[S.page.id], rows = x.draft.replies.filter((y) => y.label == null && x.suggest?.[y.reply_id] != null).map((y) => ({ reply_id: y.reply_id, label: !!x.suggest[y.reply_id] }));
    try { const r = await postJSON(`api/drafts/${encodeURIComponent(S.page.id)}/labels`, { labels: rows }); if (!r.ok) throw new Error(await errText(r)); for (const y of rows) x.draft.replies.find((z) => z.reply_id === y.reply_id).label = y.label; } catch (err) { x.err = err.message; }
    renderPage();
  } else if (t.closest("[data-draft-validate]")) {
    const x = DR[S.page.id]; x.busy = true; renderPage();
    try { const r = await postJSON(`api/drafts/${encodeURIComponent(S.page.id)}/validate`, {}), body = await r.json().catch(() => ({}));
      if (r.ok) { x.result = body; S.families = null; } else if (body.registered === false) x.result = body; else if (body.detail && typeof body.detail === "object") x.result = body.detail; else throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${r.status}`);
      x.err = "";
    } catch (err) { x.err = err.message; }
    x.busy = false; renderPage();
  } else if (t.closest("[data-verify]")) { actVerify(t.closest("[data-verify]").dataset.verify);
  } else if (t.closest("[data-del]")) { const id = t.closest("[data-del]").dataset.del; actDelete(id, behaviors().all.find((x) => x.id === id)?.name);
  } else if (t.closest("[data-targets]")) {
    const id = t.closest("[data-targets]").dataset.targets, a = ACT[id];
    try { a.targets = { list: (await getJSON("api/install/targets")).targets }; } catch (err) { a.targets = { err: `The targets did not load (${err.message}).` }; }
    renderPage();
  } else if (t.closest("[data-install]")) {
    const btn = t.closest("[data-install]"), id = btn.dataset.id, target = btn.dataset.install, a = ACT[id], ref = $(`mref-${target}`)?.value.trim();
    if (!ref) { a.install = { err: "Give the model reference first." }; return renderPage(); }
    a.install = { busy: true }; renderPage();
    try { const r = await postJSON("api/install", { trait_id: id, target, model_ref: ref }); if (!r.ok) throw new Error(await errText(r)); a.install = { res: await r.json() }; } catch (err) { a.install = { err: err.message }; }
    renderPage();
  } else if (t.closest("[data-iverify]")) {
    const btn = t.closest("[data-iverify]"), id = btn.dataset.id, a = ACT[id]; a.iverify = { busy: true }; renderPage();
    try { const r = await postJSON(`api/install/${encodeURIComponent(btn.dataset.iverify)}/verify`, { prompt: "", judge: "local", judged_n: 16 }); if (!r.ok) throw new Error(await errText(r)); a.iverify = { res: await r.json() }; } catch (err) { a.iverify = { err: err.message }; }
    renderPage();
  } else if (t.closest("[data-copy-el]")) {
    try { await navigator.clipboard.writeText($(t.closest("[data-copy-el]").dataset.copyEl).textContent); $("live").textContent = "Copied."; } catch { $("live").textContent = "Copy failed. Select the text and copy it by hand."; }
  } else if (t.closest("#chatSend")) { chatSend();
  } else if (t.closest("#chatNew")) { CH.sid = null; CH.turns = []; CH.err = ""; renderPage();
  } else if (t.closest("[data-exq]")) {
    const id = S.page.id, R = reports[id]; if (!R?.pareto) return;
    exState(R, id).qi = Number(t.closest("[data-exq]").dataset.exq); setExDose(id, exState(R, id).d);
    $("exOut")?.querySelector(`[data-exq="${exState(R, id).qi}"]`)?.focus();
  } else if (t.closest("[data-explain]")) {
    const id = t.closest("[data-explain]").dataset.explain; explains[id] = { busy: true }; renderPage();
    try { const r = await postJSON(`api/traits/${encodeURIComponent(id)}/explain`, {}), body = await r.json();
      explains[id] = r.ok ? { text: body.explanation, cached: body.cached } : { err: typeof body.detail === "string" ? body.detail : `HTTP ${r.status}` };
    } catch (err) { explains[id] = { err: `The request failed: ${err.message}` }; }
    renderPage();
  } else if (t.closest("[data-fu]")) {
    const btn = t.closest("[data-fu]"), kind = btn.dataset.fu, id = btn.dataset.id; btn.disabled = true;
    try {
      const r = await postJSON(`api/traits/${encodeURIComponent(id)}/${kind}`, kind === "repair" ? { method: "gradient" } : {}), body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${r.status}`);
      $("live").textContent = `${kind === "pareto" ? "Prompt comparison" : "Repair"} started.`;
    } catch (err) { $("live").textContent = `It did not start: ${err.message}`; alertMsg(btn, err.message); return; }
    delete reports[id]; renderPage();
  } else if (t.closest("[data-fu-stop]")) {
    const btn = t.closest("[data-fu-stop]"), id = btn.dataset.id; btn.disabled = true;
    try { await postJSON(`api/traits/${encodeURIComponent(id)}/${btn.dataset.fuStop}/stop`, {}); } catch { /* the next poll shows the state */ }
    delete reports[id]; renderPage();
  } else if (t.closest("#stopCreate")) {
    const j = S.jobs.find((x) => x.id === S.page.id), late = j && STAGES.findIndex(([k]) => k === j.stage) >= 3;
    if (late && !(await confirmDlg("Stop during the final check?", "The final check uses this version's held-out prompts once. If you stop now, they count as used, and this version cannot be checked again.", "Stop Create"))) return;
    try { const r = await postJSON(`api/create/${encodeURIComponent(S.page.id)}/stop`, {}); if (r.ok) Object.assign(j, await r.json()); } catch { /* the next poll shows the state */ }
    renderPage();
  }
});
async function loadBehaviors() {
  if (!S.api) return;
  try { S.jobs = (await getJSON("api/jobs")).jobs; } catch { S.jobs = []; }
  try { S.traits = (await getJSON(`api/traits?model_id=${encodeURIComponent(S.model)}`)).traits; } catch { S.traits = null; }
}
setInterval(async () => {
  if (!S.api || S.page?.kind !== "agent" || AG.runs[S.page.id]?.status !== "running") return;
  await renderPage();
  if (AG.runs[S.page.id]?.status !== "running") { AG.list = null; await loadBehaviors(); renderShelf(); }
}, 2000);
// A follow-up (prompt comparison, repair) for the open evidence page is queued or running.
const fuLive = () => { const R = S.page?.kind === "behavior" && reports[S.page.id]; return !!R && Object.values(R.followups || {}).some((st) => st && (st.status === "running" || st.status === "queued")); };
const alertMsg = (btn, msg) => { const p = document.createElement("p"); p.className = "meta"; p.setAttribute("role", "alert"); p.style.color = "var(--error)"; p.textContent = `It did not start: ${msg}`; btn.closest(".row2")?.after(p); btn.disabled = false; };
setInterval(async () => {
  if (!S.api || !fuLive()) return;
  const id = S.page.id; delete reports[id]; await renderPage();
  if (!fuLive()) { await loadBehaviors(); renderShelf(); }
}, 2000);
// Poll every 2 s while a Create runs; the server keeps working when the page is closed, so this also reattaches.
setInterval(async () => {
  if (!S.api || !(S.jobs.some(active) || S.page?.kind === "create")) return;
  const was = S.jobs.filter(active).map((j) => j.id).join();
  await loadBehaviors();
  if (S.jobs.filter(active).map((j) => j.id).join() !== was && !S.running) update(); else { renderShelf(); renderGuard(); }
  if (S.page?.kind === "create") renderPage();
}, 2000);

export { errText, DR, AGENT_KEY, CH, chatKey, ACT, actionsSection, quoteArg, runLine, actVerify, actDelete, versionsSection, repairSection, openPage, renderPage, loadBehaviors, fuLive, alertMsg };
