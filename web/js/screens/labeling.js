import { strip } from "../components/strip.js";
import { DR } from "./pages.js";
import { $, at, esc } from "../state.js";
function draftPage(id) {
  const x = DR[id]; if (!x?.draft) return `<p class="meta">Loading the draft.</p>`;
  const d = x.draft, p = d.parts || {}, reps = d.replies || [], n = reps.filter((r) => r.label != null).length, sug = x.suggest || {};
  const row = (r, i) => { const s = sug[r.reply_id];
    return `<li class="lab"><div><span class="meta">${i + 1}. “${esc(r.prompt)}”</span><p class="bt">${esc(r.reply)}</p>${s != null && r.label == null ? `<span class="meta">Muse suggests: ${s ? "shows it" : "does not"}</span>` : ""}</div>
      <div class="row2" role="group" aria-label="Does reply ${i + 1} show the behavior?"><button class="btn btn-sm" type="button" data-lab="${esc(r.reply_id)}" data-v="1" aria-pressed="${r.label === true}">Shows it</button><button class="btn btn-sm" type="button" data-lab="${esc(r.reply_id)}" data-v="0" aria-pressed="${r.label === false}">Does not</button></div></li>`; };
  const v = x.result;
  return `<p class="lead">“${esc(d.description)}”</p>
    <div class="kv2"><span>Steered prompt</span><span>${esc(p.pos || "")}</span><span>Neutral prompt</span><span>${esc(p.neg || "")}</span><span>Judge asks</span><span>“${esc(p.judge || "")}”</span></div>
    <h2 class="sec">Label the replies</h2><p class="lead">Does each reply show the behavior? Half the replies come from the steered prompt and half from the neutral one; the page does not say which. <span class="mono">${n}</span> of <span class="mono">${reps.length}</span> labeled.</p>
    <div class="row2"><button class="btn" type="button" data-draft-suggest="${esc(id)}" ${x.busy ? "disabled" : ""}>Ask Muse for suggestions</button>${Object.keys(sug).length ? `<button class="btn" type="button" data-draft-accept="${esc(id)}">Accept the remaining suggestions</button>` : ""}<span class="meta">Suggestions are not saved until you accept them.</span></div>
    <ol class="ruled labels">${reps.map(row).join("")}</ol>
    <div class="row2"><button class="btn" type="button" data-draft-validate="${esc(id)}" ${n < reps.length || x.busy ? "disabled" : ""}>Check the judge</button><span class="meta">${n < reps.length ? "Label every reply first." : "The judge scores the same replies. It must agree with you on at least 85%."}</span></div>
    ${x.err ? `<p class="meta" role="alert" style="color:var(--error)">${esc(x.err)}</p>` : ""}
    ${v ? v.registered ? `<p><span class="state ok">REGISTERED</span> The judge agreed on <span class="mono">${(v.agreement ?? 0).toFixed(2)}</span>. Family <span class="mono">${esc(v.family)}</span> is ready.</p><div class="row2"><button class="btn" type="button" data-create="${esc(v.family)}">Create it now</button></div>`
      : `<p><span class="state warn">BELOW THE GATE</span> The judge agreed on <span class="mono">${(v.agreement ?? 0).toFixed(2)}</span> (gate ${(v.gate ?? 0.85).toFixed(2)}). ${v.disagreements?.length ? `It disagreed on ${v.disagreements.length} repl${v.disagreements.length === 1 ? "y" : "ies"}.` : ""} Try a clearer description, or check your labels.</p>` : ""}`;
}

// Chat: the sentence's edits stay on across turns; a "read" behavior draws its channel strip under every reply.

export { draftPage };
