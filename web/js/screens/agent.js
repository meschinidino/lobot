import { mins } from "./evidence.js";
import { run, stop } from "./run.js";
import { AGENT_KEY } from "./pages.js";
import { $, esc } from "../state.js";
import { getJSON } from "../api.js";
const AG = { list: null, runs: {}, err: "" }; // agent runs (list and per run status)
async function loadAgents() { try { AG.list = (await getJSON("api/agents")).agents; } catch { AG.list = []; } }
function agentList() {
  if (!AG.list?.length) return "";
  return `<h3 class="sub">Muse runs</h3><ol class="ruled">${AG.list.map((a) => { const k = AGENT_KEY[a.status] || ["info", String(a.status || "").toUpperCase()];
    return `<li><div><button class="link" type="button" data-agent="${esc(a.run_id)}">${esc(a.description || a.run_id)}</button><br><span class="meta">${a.family ? `family ${esc(a.family)}` : "no family yet"}${a.job_id ? " · patch on the shelf" : ""}</span></div><span class="state ${k[0]}">${k[1]}</span></li>`; }).join("")}</ol>`;
}
function stepRow(st) {
  const rk = st.round ? `<span class="meta">round ${st.round}</span> ` : "";
  switch (st.step) {
    case "draft": { const p = st.parts || {};
      return [`Muse drafted the behavior`, `${rk}“${esc(st.description || "")}”<div class="kv2 tight" style="margin-top:8px"><span>Steered prompt</span><span>${esc(p.pos || "")}</span><span>Neutral prompt</span><span>${esc(p.neg || "")}</span><span>Judge asks</span><span>“${esc(p.judge || "")}”</span><span>Prompt baselines</span><span>${(p.baselines || []).length}</span></div>`, ["ok", "DONE"]]; }
    case "labels": return [`${st.by === "muse" ? "Muse" : "You"} labeled ${st.n} replies`, `${rk}${st.missing?.length ? `${st.missing.length} not labeled` : "every reply labeled"}`, ["ok", "DONE"]];
    case "validate": return [`The local judge checked the labels`, `${rk}agreement <span class="mono">${(st.agreement ?? 0).toFixed(2)}</span>, gate <span class="mono">${(st.gate ?? 0.85).toFixed(2)}</span>`, st.passed ? ["ok", "PASS"] : ["warn", "FAIL"]];
    case "register": return [`Registered the family`, `<span class="mono">${esc(st.family || "")}</span>`, ["ok", "DONE"]];
    case "create": return [`Create ran`, `${rk}${esc(st.family || "")} · ${st.seconds ? mins(st.seconds) : ""}${st.unavailable ? ` · ${esc(st.unavailable)}` : ""}`, st.verdict === "knob" ? ["ok", "KNOB"] : st.verdict === "blocked" ? ["err", "GUARD"] : ["warn", "NO KNOB"]];
    case "explain": return [`Muse explained the report`, `<span class="explain-s">${esc(st.text || "").replace(/\n/g, "<br>")}</span>`, ["ok", "DONE"]];
    case "explain_failed": return [`Muse could not explain the report`, esc(st.error || ""), ["warn", "SKIPPED"]];
    case "decide": return [st.action === "redraft" ? "Muse rewrote the description" : "The agent stopped", `${rk}${esc(st.reason || st.why || "")}${st.description ? `<br>new description: “${esc(st.description)}”` : ""}`, st.action === "redraft" ? ["info", "REDRAFT"] : ["warn", "STOP"]];
    case "refused": return [`The description was refused`, esc(st.reason || ""), ["err", "REFUSED"]];
    default: return [esc(st.step), "", ["info", "STEP"]];
  }
}
function agentPage(id) {
  const a = AG.runs[id]; if (!a) return `<p class="meta">Loading the run.</p>`;
  const k = AGENT_KEY[a.status] || ["info", String(a.status).toUpperCase()], live = a.status === "running", res = a.result || {};
  const rows = (a.steps || []).map((st) => { const [t, d, key] = stepRow(st); return `<li><div><b>${t}</b><br><span class="meta">${d}</span></div><span class="state ${key[0]}">${key[1]}</span></li>`; }).join("");
  return `<p><span class="state ${k[0]}">${k[1]}</span> <span class="meta">${esc(res.description || a.description || "")}</span></p>
    ${live ? `<div class="row2"><button class="btn" type="button" data-agent-stop="${esc(id)}">Stop</button><span class="meta">The run stops between steps. The server keeps working if you close this page.</span></div>` : ""}
    <ol class="ruled stages agent">${rows || `<li><div><b>Starting</b><br><span class="meta">Muse is drafting the behavior.</span></div><span class="state info">NOW</span></li>`}${live && rows ? `<li class="wait"><div><b>Working</b><br><span class="meta">next step in progress</span></div><span class="state info">NOW</span></li>` : ""}</ol>
    ${a.error ? `<p class="empty" style="color:var(--error)">${esc(a.error)}</p>` : ""}
    ${a.job_id ? `<div class="row2"><button class="btn" type="button" data-beh="${esc(a.job_id)}">Open evidence</button><span class="meta">The patch is on the shelf.</span></div>` : ""}
    ${res.muse?.usage ? `<p class="meta">Muse usage: ${res.muse.usage.calls} calls, ${res.muse.usage.input_tokens} input and ${res.muse.usage.output_tokens} output tokens (${esc(res.muse.model || "")}).</p>` : ""}`;
}

export { AG, loadAgents, agentList, stepRow, agentPage };
