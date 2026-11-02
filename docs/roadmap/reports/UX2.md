# UX2 report: the 100X features on screen

Builds on [UX-100x.md](UX-100x.md). Design entries: docs/DESIGN.md
("State keys explained", "New behavior from words", "Chat with these
edits", "Use it elsewhere"; decisions log 2026-11-02).

## Results first

- **Every 100X backend feature now has a screen:** describe a behavior
  in words (Muse agent or self-labeling), chat with live dials,
  re-check on this machine, remove the behavior on one answer, and
  install into llama.cpp or Ollama.
- **Most of it ran on real Llama 3.2 3B weights in the browser** (live
  server, llama.cpp lane for Create). The table below says what ran live
  and what did not.
- **A real backend bug was found and fixed.** A client that left in the
  middle of an NDJSON stream kept the engine lock until a restart. Now
  the lock frees once the running step ends (measured: about 100 s after
  a disconnect during the guard step; before the fix it never freed).
- **The state keys now have explanations** (user report): a legend on
  the Library and evidence pages and hover text on the shelf. A failed
  turn-up card now shows INCREASE FAIL; before, cowboy v2 showed
  DECREASE FAIL although its Increase card failed too.

## What ran on real weights

| Screen | Result |
|---|---|
| Chat, pirate v4 +0.50 | The guard blocked it (refusal at 17% of its usual level on a forgery request). Shown as a blocked turn. |
| Chat, cowboy v2 +0.50, reading cowboy | Reply streamed after the first-turn guard check (about 4 min); the read dial drew under every word. The reading sits high because it reads the pushed direction; the page now says so. |
| Re-check, cowboy v2 | HOLDS: measured +0.17 on 16 held-out prompts, shipped +0.18 [+0.07, +0.30]. |
| Remove, cowboy v2 and pirate v4 | The guard blocked both removals in the full-answer check (a test prompt lost its refusal). Message fixed to name the removal. |
| Muse agent, "speaks like a nervous museum tour guide" | Three rounds; the local judge agreed with Muse's labels at 0.30, 0.40, 0.50 (gate 0.85); Muse rewrote the description twice, then stopped. 8 Muse calls, 5,208 input and 9,172 output tokens. Every step showed live. |
| Muse agent, past run from disk (sports commentator) | Listed, opened, ends with Open evidence; the patch is on the shelf (registered as a job). |
| Label it yourself, "speaks like a TV weather reporter" | The local model drafted the parts and 20 replies; one label saved. The local drafter drifted ("is formal and professional"), which is why the Muse path exists. |
| Install, pirate v4 into llama.cpp (Q4_K_M) | Installed; the run line is a copyable shell command. |
| Check the installed patch | **FAILED**: "Unpatched run failed: llama-cli timed out". The install route works; the llama-cli run inside the check does not finish. Not investigated yet (IN backend). |

Not run live:
- **The two-column removal view.** Both live removals were guard-blocked.
  Research mode would skip the guard; the agent's permission check
  refused to turn the guard off, so that view is untested.
- **Muse suggestions on the labeling page** and **Check the judge**
  (needs all 20 labels; labeling is the user's judgment).
- **Ollama install** (Ollama is not installed on this Mac).

## What changed

- `web/index.html`: New behavior page (describe box, two ways, Muse
  runs), agent run page, labeling page, Chat page, "Use it elsewhere"
  on the evidence page, key legend and hover text, INCREASE FAIL.
- `server.py`: `GET /api/agents`; `GET /api/agent/{id}` reads finished
  runs from disk; an agent's Create run is registered as a job;
  `_ndjson` closes a stream's generator when the client leaves.
- `job_import.py`: the run import, shared by `scripts/import_run.py` and
  the agent.

## Tests

- `tests/test_followups.py`: agent listing and registration; a
  disconnected stream releases its lock and sets cancel; a normal end
  does not cancel the next run.
- Full suite: 695 tests pass, 5 skipped (before the last UI-only edits).

## Known gaps

- The first chat turn and every removal wait for a full-generation
  guard (minutes). The guard is cached per edit set for chat.
- A disconnect frees the lock only after the running step; a guard step
  does not check the cancel event.
- The model choice is not persistent: the server boots on Qwen, so each
  live session starts with a switch to Llama (open TODO).
