import { legend } from "../components/legend.js";
import { SCREEN, SHELF_KEYS, VERDICT, short, stroke } from "./evidence.js";
import { behaviors, update } from "./workbench.js";
import { $, S, at, esc, fd } from "../state.js";
function libraryPage() {
  const all = behaviors().all, fams = [...new Set(all.map((b) => b.name))];
  if (!all.length) return `<p class="empty">No behaviors for this model yet.</p>`;
  const newest = fams.map((f) => all.filter((b) => b.name === f).sort((a, b) => b.version - a.version)[0]).sort((a, b) => (b.created || 0) - (a.created || 0));
  const cnt = (v) => newest.filter((b) => (b.ev || {}).pareto_verdict === v).length, none = newest.filter((b) => !(b.ev || {}).pareto_verdict).length;
  const verbs = (b) => [["Increase", "update", b.cards.update_up], ["Decrease", "update", b.cards.update_down], ["Delete", "delete", b.cards.delete]]
    .map(([l, k, c]) => `<span class="vb">${stroke(k)}<span>${l}</span> <span class="state ${!c ? "info" : c.passed ? "ok" : "warn"}">${!c ? "NOT TESTED" : c.passed ? "PASS" : "FAIL"}</span></span>`).join("");
  const screens = (e) => Object.entries(e.screens || {}).map(([n, s]) => `<span class="state ${s === "blocked" ? "err" : s === "would_block" ? "warn" : "ok"}">${(SCREEN[n] || n).toUpperCase()} ${s === "blocked" ? "BLOCKED" : s === "would_block" ? "FLAGGED" : "OK"}</span>`).join(" ");
  const rows = newest.map((b) => { const e = b.ev || {}, vk = VERDICT[e.pareto_verdict], older = all.filter((x) => x.name === b.name && x !== b).sort((x, y) => y.version - x.version);
    return `<li><div class="lib">
      <div><button class="link" type="button" data-beh="${esc(b.id)}"><b class="name">${esc(b.name)}</b></button> <span class="meta">v${b.version}${b.dose != null ? ` · layer ${b.layer} · dose ${fd(b.dose)}` : ""}</span>
        ${older.length ? `<br><span class="meta">older: ${older.map((o) => `<button class="link" type="button" data-beh="${esc(o.id)}">v${o.version}</button>`).join(" ")}</span>` : ""}</div>
      <div class="vbs">${verbs(b)}</div>
      <div class="kv2 tight"><span>Against a prompt</span><span>${vk ? `<span class="state ${vk[0]}">${vk[1]}</span>` : `<span class="meta">not compared yet</span>`}</span>
        <span>Fluency at its dose</span><span class="mono">${e.fluency != null ? `${e.fluency.toFixed(2)}${e.fluency < 0.5 ? ' <span class="state warn">NOT FLUENT</span>' : ""}` : '<span class="meta">not measured</span>'}${e.dose_rule ? "" : b.dose != null ? ' <span class="meta">chosen before the fluency check</span>' : ""}</span>
        <span>Guard</span><span>${screens(e) || '<span class="meta">not recorded</span>'}</span>
        ${e.repair ? `<span>Repair</span><span>${Object.entries(e.repair).map(([m, s]) => `<span class="state ${s === "repaired" ? "ok" : "warn"}">${m.toUpperCase()} ${String(s).toUpperCase()}</span>`).join(" ")}</span>` : ""}</div>
    </div><span class="state ${b.kind}">${b.key}</span></li>`; }).join("");
  return `<p class="lead">${newest.length} behavior${newest.length === 1 ? "" : "s"} on <span class="mono">${esc(short(S.model))}</span>, newest version of each. Against a plain prompt: ${cnt("vector_dominates")} beat it on effect and side effects, ${cnt("tradeoff")} trade off, ${cnt("prompt_dominates")} lose to it, and ${none} ${none === 1 ? "has" : "have"} no comparison yet.</p>
    ${legend([...SHELF_KEYS, "KNOB BETTER", "NO CLEAR WINNER", "PROMPT BETTER", "NOT FLUENT", "FLAGGED"], true)}
    <ol class="ruled library">${rows}</ol>`;
}

export { libraryPage };
