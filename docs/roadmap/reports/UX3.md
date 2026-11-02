# UX3 report: the last screens on real weights, and docs

Closes the "not run live" list of [UX2.md](UX2.md). All runs: Llama 3.2
3B, live server, browser pane.

## Results first

- **Every screen has now run on real weights.** Nothing in the app is
  left at "tested with fakes only", except Ollama (not installed).
- **A bug in the removal screen was found and fixed.** The removal
  stream ends with `complete`, not `done`; the page waited for `done`,
  so a finished removal read as "The run stopped before it finished".
  `docs/reference/api.md` said `done` too; it now says `complete`.
- **Removing noir changes the answer but does not remove the style.**
  That matches noir's failed Delete card.
- **Muse suggestions and the judge check work.** On a draft the local
  model wrote ("TV weather reporter"), the judge agreed with Muse's
  labels on 0.25 (gate 0.85), so nothing registered.

## Live results

| Screen | Result |
|---|---|
| Remove, noir, "Tell me about your favorite food." | Guard passed (verified; refusal kept at 90%). Both answers identical: the plain answer has no noir style, so removal changes nothing. The page now says so (NO CHANGE) and suggests a prompt that asks for the behavior. |
| Remove, noir, "Speak like a hard-boiled noir detective. Tell me about your favorite food." | The answer changed (as is: "Listen, pal ... a dame with a heart of gold"; removed: "(sighing) ... as dark as a sinner's soul"). The noir style stays. |
| Label it yourself, "speaks like a TV weather reporter" | The local drafter wrote "Deliver the forecast in a formal, professional tone" and judge "is formal and professional". Ask Muse for suggestions: 20 suggestions shown, none saved. Accept: 20 of 20 labeled. Check the judge: BELOW THE GATE, agreement 0.25, 15 disagreements. Nothing registered; `store/` unchanged. |

Not run: Ollama install (not installed on this Mac).

## Docs changed

- New user guide [docs/guides/app.md](../../guides/app.md): every
  screen, the state keys, and the warnings (weights, one job at a time,
  the guard).
- `docs/README.md`, `docs/guides/create.md`, `docs/guides/share.md`:
  point to the screens; measured agent and install results.
- `AGENTS.md`: map rows for `followups.py` and `job_import.py`; the
  import command; "Backend rules learned the hard way" (streams,
  subprocess stdin, jobs, never turn the guard off); "Checking UI work".
- `CLAUDE.md`: points to `docs/guides/app.md` and adds "Checking work".
- `docs/agent-boot.md`: switching to Llama, Muse key, llama lane,
  guard waits, the import command, current suite count.
- `docs/reference/api.md`: the removal stream ends with `complete`.

## Tests

Full suite: see the commit message (fakes only; the UI changes are
covered by the live runs above).
