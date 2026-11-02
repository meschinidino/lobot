import { AG, agentList } from "./agent.js";
import { noWeights } from "./evidence.js";
import { creating } from "./run.js";
import { $, S, at, esc } from "../state.js";
function describeBox() {
  const off = noWeights() || (S.state?.no_model ? "This server runs without weights." : "");
  return `<h2 class="sec" style="margin-top:24px">Describe a behavior</h2>
    <p class="lead">Write the behavior in plain words. Two ways to build it, and both end in the same held-out report card:</p>
    <label class="label" for="descIn" style="margin-top:12px">Behavior</label><textarea id="descIn" class="field" rows="2" spellcheck="true" placeholder="for example: talks like a nervous museum tour guide">${esc(S.descDraft || "")}</textarea>
    <ol class="ruled ways"><li><div><b>Let Muse build it.</b><br><span class="meta">Muse Spark drafts the contrast prompts and the judge question, the model writes 20 replies, Muse labels them, and the local judge must agree on at least 85% before Create runs. If a gate fails, Muse may rewrite the description, up to 3 rounds. Needs META_API_KEY on the server.</span></div><button class="btn" type="button" data-agent-start ${off || creating() ? "disabled" : ""}>Let Muse build it</button></li>
      <li><div><b>Label it yourself.</b><br><span class="meta">The loaded model drafts the parts and writes 20 replies. You mark each one: does it show the behavior? The judge must agree with you on at least 85%. Then you start Create.</span></div><button class="btn" type="button" data-draft-start ${off ? "disabled" : ""}>Label it yourself</button></li></ol>
    <label class="meta" style="display:flex;gap:8px;align-items:center;margin-top:8px"><input type="checkbox" id="laneIn" ${S.agentLane !== "torch" ? "checked" : ""}> Run Create on the llama.cpp lane (about 8 to 13 min instead of about 35)</label>
    <p class="meta" role="alert" id="descErr" style="color:var(--error)">${esc(off ? `${off} Start the server with weights to build a behavior.` : AG.err || "")}</p>
    ${agentList()}`;
}

export { describeBox };
