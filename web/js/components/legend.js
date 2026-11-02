import { KEY_HELP } from "../screens/evidence.js";
import { $, esc } from "../state.js";
function legend(keys, open = false) {
  return `<details class="legend"${open ? " open" : ""}><summary>What the keys mean</summary><dl>${keys.map((k) => `<dt><span class="state ${k === "PASS" || k === "KNOB BETTER" || k === "OK" ? "ok" : k === "NO CLEAR WINNER" || k === "NOT TESTED" ? "info" : "warn"}">${k}</span></dt><dd>${esc(KEY_HELP[k])}</dd>`).join("")}</dl></details>`;
}
// Library: every behavior on this model, newest version first, with where it stands against its own prompt.

export { legend };
