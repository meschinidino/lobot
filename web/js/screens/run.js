import { refreshState, update } from "./workbench.js";
import { NST, STAGES, active, closeInspector, closePop, drawN, inspect, pop, proveLift, renderInspector, renderTree, treeRuns } from "./workbench.js";
import { $, S, addResult, at, baselineOnly, curExp, editsText, isB, rowsKey } from "../state.js";
import { postJSON } from "../api.js";
/* ---------- run ---------- */
function setStatus(kind, key, msg) { $("runState").className = `state ${kind}`; $("runState").textContent = key; $("runMsg").textContent = msg; }
function setRunning(on) {
  S.running = on;
  const had = document.activeElement === $(on ? "runBtn" : "stopBtn");
  $("runBtn").hidden = on; $("stopBtn").hidden = !on; $("stopBtn").disabled = false; $("fanBtn").disabled = on;
  if (had) $(on ? "stopBtn" : "runBtn").focus();
  $("prompt").readOnly = on;
  update();
}
// Reads an NDJSON stream; onEvent returns true when the stream is finished for the caller.
async function stream(url, body, onEvent) {
  const r = await postJSON(url, body);
  if (!r.ok) { const b = await r.json().catch(() => ({})); throw new Error(typeof b.detail === "string" ? b.detail : `The server returned HTTP ${r.status}.`); }
  const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "", done = false;
  while (!done) {
    const c = await reader.read(); done = c.done; buf += dec.decode(c.value || new Uint8Array(), { stream: !done });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const ev = JSON.parse(buf.slice(0, nl)); buf = buf.slice(nl + 1);
      if (ev.stage === "error") throw new Error(ev.message);
      if (onEvent(ev)) return true;
    }
  }
  return false;
}
const setupKey = () => `${S.model}|${rowsKey(S.rows)}`;
const creating = () => S.jobs.find(active);
// what: the words for the edit set the guard checked (default: the sentence's edits).
function blockedMsg(g, what = editsText(S.rows)) {
  const pct = (x) => Math.round(x * 100) + "%", where = g.display_prompt ? ` on “${g.display_prompt}”` : "";
  // The first-token check passed, so the block came from the full-answer confirm step.
  if (g.min_ratio != null && g.min_ratio >= (g.limit ?? 0.5))
    return `The refusal guard blocked ${what}. The first-token check passed (refusal at ${pct(g.min_ratio)} of its usual level; the limit is ${pct(g.limit ?? 0.5)}), but in the full-answer check ${g.blocked_prompts?.length > 1 ? `${g.blocked_prompts.length} test prompts` : "a test prompt"} no longer got a refusal. Lower the dose, or use Research mode.`;
  return `The refusal guard blocked ${what}. With these edits, the model's chance of refusing fell to ${g.min_ratio != null ? pct(g.min_ratio) : "under half"} of its usual level${where}; the limit is ${pct(g.limit ?? 0.5)}. Lower the dose, or use Research mode.`;
}
async function run(fan = false) {
  if (S.running) return;
  const prompt = $("prompt").value.trim(), job = creating();
  if (!S.api) return setStatus("err", "OFFLINE", "The lobot server is not reachable. Start it with: uv run python server.py");
  if (job) return setStatus("info", "CREATE RUNNING", `Run is off while Create builds ${job.trait} (stage ${STAGES.findIndex(([k]) => k === job.stage) + 1} of ${NST}). Open it on the shelf to stop it.`);
  if (!prompt) { $("prompt").focus(); return setStatus("err", "ERROR", "Type a prompt first."); }
  const beh = S.rows.find(isB);
  if (fan && !beh) return;
  closePop(); setRunning(true);
  const key = setupKey(), research = S.research, guardBody = { guard: !research, research_mode: research };
  const t0 = performance.now(), msgs = { weights: "Loading weights and calibration.", guard: "Checking the refusal guard.", baseline: "Baseline pass.", steered: "Edited pass.", columns: "Generating every dose at once." };
  const tick = setInterval(() => ($("runMsg").textContent = `${$("runMsg").textContent.split(" · ")[0]} · ${((performance.now() - t0) / 1000).toFixed(1)} s`), 100);
  setStatus("info", "RUNNING", msgs.weights);
  const end = () => { clearInterval(tick); S.live = null; S.checking = false; setRunning(false); refreshState(); };
  const onCommon = (ev) => {
    if (ev.stage === "weights") { S.checking = !research; renderGuard(); }
    if (ev.stage === "guard" || ev.stage === "columns" && !research) { S.checking = false; if (!research) S.guardSeen[key] = "passed"; renderGuard(); }
    if (ev.stage === "stopped") { end(); setStatus("info", "STOPPED", "Stopped. The tree keeps the runs it had."); return true; }
    if (ev.stage === "blocked") {
      S.guardSeen[key] = "blocked"; end(); setStatus("err", "BLOCKED", blockedMsg(ev.guard || {}));
      $("live").textContent = "Blocked by the refusal guard."; return true;
    }
    if (msgs[ev.stage]) setStatus("info", "RUNNING", msgs[ev.stage]);
    return false;
  };
  const after = (n, g) => {
    if (g && g.would_block && !g.enforced && !research) $("runMsg").textContent += " The refusal guard would block these edits.";
    const T = treeRuns(), got = T && T.runs.find((x) => x.n === n);
    if (got) $("live").textContent = got.split ? `Run ${n} complete. It leaves run 1 at ${at(got.split)}.` : `Run ${n} complete. It matches run 1 token for token.`;
    $("runs").querySelector(`[data-pick="${n}"]`)?.focus({ preventScroll: false });
  };
  try {
    let finished;
    if (beh) {
      // A behavior runs through /api/update: dose 0 is the baseline column, the others are branches.
      const doses = fan ? [0, ...[1, 2, 3, 4, 5].map((i) => Math.round(beh.dose * i / 5 * 100) / 100)] : [0, beh.dose];
      finished = await stream("api/update", { prompt, trait: beh.trait, doses, ...guardBody }, (ev) => {
        if (onCommon(ev)) return true;
        if (ev.stage === "columns") { S.live = { trait: beh.trait, cols: ev.doses.map((d) => ({ dose: d, pieces: [] })) }; renderTree(); }
        else if (ev.stage === "token" && S.live) { S.live.cols[ev.col].pieces.push(ev.text); renderTree(); }
        else if (ev.stage === "complete") {
          const res = ev.result, cols = S.live ? S.live.cols : res.columns.map((c) => ({ dose: c.dose, pieces: [c.text] }));
          const bi = res.columns.findIndex((c) => c.dose === 0), base = res.columns[bi];
          const e = curExp(), differs = e && e.baseline.text !== base.text;
          let last = 0;
          res.columns.forEach((c, i) => {
            if (i === bi) return;
            last = addResult({ model_id: res.model_id, prompt: res.prompt, rows: [{ trait: beh.trait, dose: c.dose }], baseline: { text: base.text, tokens: base.tokens },
              steered: { text: c.text, tokens: c.tokens }, split: splitOf(cols[bi].pieces, cols[i].pieces, base.text, c.text, res.fallback_used),
              seconds: res.seconds, recipe: `update trait=${beh.trait} layer=${res.layer} dose=${c.dose} scale=${res.scale}`, post: "", research }, true);
          });
          drawN = last; end();
          if (differs) $("runMsg").textContent = "Note: the batched baseline differs from this experiment's run 1. Splits are measured against the batched baseline.";
          after(last, res.guard); return true;
        }
        return false;
      });
    } else {
      // With no edit, one neuron at dose 0 (exactly no change) gets the baseline; only run 1 is kept.
      const read = S.read, toks = [], none = !S.rows.length, p0 = (S.state?.preset_rows || [{ layer: 0, neuron: 0 }])[0];
      const rows = none ? [{ layer: p0.layer, neuron: p0.neuron, dose: 0 }] : S.rows;
      finished = await stream("api/run", { prompt, rows, ...guardBody, dial_trait: read }, (ev) => {
        if (onCommon(ev)) return true;
        if (ev.stage === "token") toks.push({ text: ev.text, value: ev.dial && read in ev.dial ? ev.dial[read] : null });
        if (ev.stage === "complete") {
          const d = ev.result.dial, dial = d && { trait: d.trait, steered: d.steered || toks, baseline: d.baseline || null };
          if (none) { baselineOnly({ ...ev.result, dial }); end(); setStatus("info", "NO EDIT", "Run 1 is the model with no edit. Add an edit and run again to fork it."); return true; }
          const n = addResult({ ...ev.result, research, dial }); drawN = n; end(); after(n, ev.result.guard); proveLift(n); return true;
        }
        return false;
      });
    }
    // The stream ended without a final event (server restart, dropped connection).
    if (!finished && S.running) { end(); setStatus("err", "ERROR", "The run ended early, with no result. Try again."); }
  } catch (err) {
    if (S.checking) S.guardSeen[key] = "failed";
    end(); setStatus("err", "ERROR", String(err.message || err));
  } finally { clearInterval(tick); }
}
// First differing token between the baseline column and a dose column. With the singles fallback each column is one piece, so only the character is known.
function splitOf(bp, cp, bt, ct, fallback) {
  if (bt === ct) return null;
  let c = 0; while (c < bt.length && c < ct.length && bt[c] === ct[c]) c++;
  let tok = null;
  if (!fallback) { tok = 0; let acc = 0; while (tok < bp.length && tok < cp.length && bp[tok] === cp[tok]) acc += bp[tok++].length; c = acc; }
  return { token: tok, base_char: c, steered_char: c };
}
async function stop() {
  if (!S.running || $("stopBtn").disabled) return;
  $("stopBtn").disabled = true; setStatus("info", "STOPPING", "Stops after the current token.");
  try { await fetch("api/stop", { method: "POST" }); } catch { setStatus("err", "ERROR", "The stop request did not reach the server."); $("stopBtn").disabled = false; }
}
$("runBtn").addEventListener("click", () => run());
$("fanBtn").addEventListener("click", () => run(true));
$("stopBtn").addEventListener("click", stop);
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
  else if (e.key === "Escape") { if (pop) closePop(); else if (S.running) stop(); else if (S.card && S.inspect) { S.card = null; renderInspector(); } else if (S.inspect || S.card) closeInspector(); }
});

/* ---------- refusal guard key and research mode (design decision 6A) ---------- */
function renderGuard() {
  const seen = S.guardSeen[setupKey()];
  document.querySelectorAll("[data-research]").forEach((x) => (x.hidden = !S.research));
  $("researchBtn").hidden = S.research || !S.api;
  const [kind, text] = S.research ? ["warn", "RESEARCH MODE · GUARD OFF"] : S.checking ? ["info", "CHECKING EDITS"] : !S.rows.length ? ["info", "GUARD ON"]
    : seen === "passed" ? ["ok", "CHECK PASSED"] : seen === "blocked" ? ["err", "BLOCKED"] : seen === "failed" ? ["err", "CHECK FAILED"] : ["info", "GUARD ON · NOT CHECKED"];
  $("guardKey").className = `state ${kind}`; $("guardKey").textContent = text;
  const beh = S.rows.find(isB), job = creating();
  $("fanBtn").hidden = !beh; $("fanBtn").disabled = S.running || !beh || beh.dose === 0;
  $("runBtn").disabled = !S.api || !!job || S.reading > 0;
  $("findForm").querySelector("button").disabled = !S.api || S.running || S.reading > 0;
  if (S.reading > 0 && !S.running) $("plan").textContent = "Run waits while the model reads the neurons at a word.";
  if (job && !S.running) $("plan").textContent = `Run is off: Create is building ${job.trait}, stage ${STAGES.findIndex(([k]) => k === job.stage) + 1} of ${NST}.`;
}
function confirmDlg(title, text, ok) {
  return new Promise((res) => {
    const d = $("dlg"); $("dlgH").textContent = title; $("dlgP").textContent = text; $("dlgOk").textContent = ok; d.returnValue = "";
    d.addEventListener("close", () => res(d.returnValue === "ok"), { once: true }); d.showModal();
  });
}
$("researchBtn").addEventListener("click", async () => {
  const yes = await confirmDlg("Turn the guard off?", "Research mode turns the refusal guard off until you reload this page. Edits that weaken the model's refusals will run. Every run records research mode, and the share card shows it.", "Turn the guard off");
  if (yes) { S.research = true; update(); setStatus("warn", "RESEARCH MODE", "The refusal guard is off for this session. Every run records it."); }
  $(yes ? "runBtn" : "researchBtn").focus();
});
document.querySelector("[data-guard-on]").addEventListener("click", () => { S.research = false; update(); setStatus("info", "GUARD ON", "The refusal guard is on again."); $("researchBtn").focus(); });

export { setStatus, setRunning, stream, setupKey, creating, blockedMsg, run, splitOf, stop, renderGuard, confirmDlg };
