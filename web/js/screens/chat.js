import { strip } from "../components/strip.js";
import { noWeights, short } from "./evidence.js";
import { blockedMsg, stream } from "./run.js";
import { CH, chatKey, errText, renderPage } from "./pages.js";
import { $, S, eid, esc, fd, isB } from "../state.js";
import { postJSON } from "../api.js";
function chatPage() {
  const beh = S.rows.find(isB), neu = S.rows.filter((r) => !isB(r));
  const edits = S.rows.length ? S.rows.map((r) => `<span class="mono">${esc(eid(r))} ${fd(r.dose)}</span>`).join(", ") : "no edit";
  const msgs = CH.turns.map((t, i) => t.role === "user" ? `<div class="msg user"><span class="bl mono">you</span><p class="bt">${esc(t.text)}</p></div>`
    : `<div class="msg bot${t.dial ? " dialed" : ""}" data-i="${i}"><span class="bl mono">model${t.blocked ? " · blocked by the guard" : ""}</span><p class="bt">${t.dial ? t.dial.map((x, j) => `<span class="w" data-v="${x.value}" data-i="${j}">${esc(x.text)}</span>`).join("") : esc(t.text || "")}${t.live ? '<span class="redact" style="width:3ch"></span>' : ""}</p>${t.live && !t.text && i === 1 ? `<p class="meta">Before the first reply, the guard checks these edits with full generation on every screen. That can take a few minutes; the result is kept for these edits.</p>` : ""}<svg class="cstrip" aria-hidden="true"></svg></div>`).join("");
  return `<p class="lead">On <span class="mono">${esc(short(S.model))}</span> with ${edits}${S.read ? `, reading <span class="mono">${esc(S.read)}</span>` : ""}. The edits stay on for every turn. Change them in the sentence on the tree; a change starts a new chat.${S.research ? " Research mode: the guard is off." : ""}</p>
    ${neu.length && beh ? `<p class="meta">A chat uses one behavior or neurons, not both.</p>` : ""}
    ${beh && S.read === beh.trait ? `<p class="note"><span class="state info">SAME DIRECTION</span> You read the behavior you push. The reading includes the push itself, so it sits high from the first word. Read another behavior to see a side effect.</p>` : ""}
    <div class="chat" id="chatLog" aria-live="polite">${msgs || `<p class="empty">Write a message. The model answers with the edits on${S.read ? `, and the pen line under each reply reads ${esc(S.read)} word by word (band: 0 = neutral, 1 = the behavior's mean)` : ""}.</p>`}</div>
    ${CH.err ? `<p class="meta" role="alert" style="color:var(--error)">${esc(CH.err)}</p>` : ""}
    <label class="label" for="chatIn" style="margin-top:16px">Message</label><textarea id="chatIn" class="field" rows="2" spellcheck="true" ${CH.busy ? "disabled" : ""}></textarea>
    <div class="row2"><button class="btn btn-primary" type="button" id="chatSend" ${CH.busy || noWeights() ? "disabled" : ""}>Send <kbd class="kbd">⌘ Enter</kbd></button><button class="btn" type="button" id="chatNew" ${CH.busy || !CH.turns.length ? "disabled" : ""}>New chat</button><span class="meta">${noWeights() ? `${noWeights()} Chat needs weights.` : ""}</span></div>`;
}
function drawChatStrips() {
  for (const m of document.querySelectorAll(".msg.dialed")) { const svg = m.querySelector(".cstrip"), box = m.getBoundingClientRect(); svg.setAttribute("width", box.width); svg.setAttribute("height", box.height); svg.innerHTML = strip(m.querySelector(".bt"), box, false); }
}
async function chatSend() {
  const msg = $("chatIn").value.trim(); if (!msg || CH.busy) return;
  if (CH.key !== chatKey()) { CH.sid = null; CH.turns = []; }
  CH.busy = true; CH.err = ""; CH.turns.push({ role: "user", text: msg }); const bot = { role: "bot", text: "", live: true }; CH.turns.push(bot); renderPage();
  try {
    if (!CH.sid) {
      const beh = S.rows.find(isB), r = await postJSON("api/chat/start", { rows: beh ? [] : S.rows.map((x) => ({ layer: x.layer, neuron: x.neuron, dose: x.dose })), trait: beh ? beh.trait : null, dose: beh ? beh.dose : null,
        dial_traits: S.read ? [S.read] : [], guard: !S.research, research_mode: S.research, max_new_tokens: 150 });
      if (!r.ok) throw new Error(await errText(r));
      CH.sid = (await r.json()).session_id; CH.key = chatKey();
    }
    await stream("api/chat", { session_id: CH.sid, message: msg }, (ev) => {
      if (ev.stage === "token") { bot.text += ev.text; const el = document.querySelector(`.msg[data-i="${CH.turns.length - 1}"] .bt`); if (el) el.textContent = bot.text; }
      else if (ev.stage === "blocked") { bot.blocked = true; bot.text = blockedMsg(ev.guard); }
      else if (ev.stage === "done") { const d = ev.result || {}; bot.text = d.reply ?? bot.text; const ds = d.dials && S.read ? d.dials[S.read] : null; if (ds && ds.length) bot.dial = ds; return true; }
      return false;
    });
  } catch (e) { CH.err = e.message; if (!bot.text) CH.turns.splice(-2, 2); }
  bot.live = false; CH.busy = false; renderPage();
  if (CH.err && !bot.text && $("chatIn")) $("chatIn").value = msg; // a failed send keeps the message
  $("chatIn")?.focus();
}

// Evidence actions: re-check on this machine, remove the behavior on one prompt, install into llama.cpp or Ollama.

export { chatPage, drawChatStrips, chatSend };
