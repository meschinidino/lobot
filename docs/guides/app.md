# Use the app

The app is a workbench on `http://127.0.0.1:7860`. Start it in [live
mode](live-mode.md) (real weights) or [replay mode](replay-mode.md) (no
weights, recorded runs only). This guide walks through every screen.

Warnings first:
- Everything that generates text needs weights. In replay mode, the
  buttons that need weights are off and say so.
- One model job runs at a time. A second one waits or shows "A run is
  in progress". If you close a tab during a run, the server frees the
  model after the step that is running (up to a few minutes for a guard
  check).
- The guard is on unless you turn on Research mode. Every run in
  Research mode records that.

## The workbench (first screen)

- **The sentence** sets the model, the edits and the prompt: "On
  [model], push [edit] at [dose] … and ask". "+ edit" adds a neuron or a
  saved behavior; "+ read" picks one behavior to read word by word.
- **Run** generates; every run adds a branch to the experiment tree,
  split at the first changed word. The full-answer guard check shows a
  progress bar with its screen and generation count. It batches its
  prompts, and it reuses unedited generations from earlier checks.
  A reload during a run asks first.
  If you reload anyway, the server cancels the run and keeps nothing
  from it. Results save only when a run completes.
- **The knob** sets the dose of the selected edit. It never generates.
- **Chat with these edits** opens a chat that keeps the sentence's edits
  on for every turn (see Chat).

## The shelf

- **Models**: the active model and the others on disk.
- **Behaviors**: the newest version of each behavior on this model, with
  one state key. Hover a key for its meaning. **Library** lists all of
  them; **+ New** builds a new one.

| Key | Meaning |
|---|---|
| PASS | A dose passed its checks, and no other key applies |
| INCREASE FAIL | Turning the behavior up did not pass its held-out report card |
| DECREASE FAIL | Turning it up passed; turning it down did not |
| GUARD (amber) | A guard check would block the patch; it is recorded, not enforced |
| GUARD (red) | The guard blocked the patch; it cannot be used |
| NO KNOB | No layer and dose passed; nothing to use |

- **Neurons**: find neurons that push a word. Under it, **Find
  with Muse** takes a goal and the prompt from the sentence.
  Muse Spark calls lobot's own tools: find, read, measure and
  try. Before each measure or try, Muse writes a prediction.
  The panel shows each step with CONFIRMED or NOT CONFIRMED.
  Then it shows the final recipe, the guard state and PASS or
  FAIL. "Use in the sentence" loads the recipe rows into the
  workbench. lobot decides PASS or FAIL. The guard stays on.

## Library

One row per behavior (newest version, older versions as links): the
verb results, where it stands against a plain prompt (KNOB BETTER,
NO CLEAR WINNER, PROMPT BETTER), fluency at its dose, each guard
screen, and the repair state. "What the keys mean" explains every key.

## The evidence page

Open a behavior from the shelf or the Library. Top to bottom:

1. **The verb table**: Read, Increase, Decrease, Delete, each with its
   held-out result, effect with a 95% interval, the random-direction
   control and side effects (relevance and fluency at the dose).
2. **The dose rule**: which rule chose the dose. The current rule
   (fluency-first-v3) picks the strongest dose whose answers stay
   fluent on nearly every prompt, not only on average. A dose chosen
   before the fluency check shows a warning with its fluency.
3. **Compared with a plain prompt**: a verdict sentence, a chart of
   effect against side-effect cost per dose and per prompt, and the same
   values as a table. "Compare with a prompt" runs it for an older run
   (about 2 min on the llama lane).
4. **Turn the dose**: a dial with one stop per measured dose. Each stop
   shows the recorded replies to the same question: no edit, the patch
   (first changed word in pink) and a plain prompt. It never generates.
5. **How Create chose the dose**: the layer and dose grid with each
   cell's reason (chosen, qualified, NLL too high, not fluent).
6. **Guard**: one row per screen (refusal, honesty, privacy).
7. **Repair**: shown when a screen flags. "Try repair" runs guard repair
   and lists every candidate; a repaired patch can be downloaded.
8. **Versions**: every version of the behavior.
9. **Use it elsewhere**:
   - **Re-check** re-runs 16 held-out prompts on this machine and says
     HOLDS or DOES NOT HOLD against the interval the patch shipped with.
   - **Remove it from one answer** removes the behavior's direction on
     one prompt and shows the answer as is and with it removed (newest
     version only). The guard checks the removal first (minutes). Use a
     prompt that asks for the behavior ("Speak like a noir detective.
     …"); on a plain prompt the behavior is absent, so nothing changes.
   - **Install** finds llama.cpp and Ollama, installs the patch as a
     control vector, gives a run line to copy, and checks the installed
     patch on 16 of its held-out prompts.
10. **Second opinion: Muse Spark**: Muse Spark re-scores the base
    and steered replies of the turn-up pairs. The card sits next to
    the turn-up card. Press "Run second opinion (about 3 min, 600
    Meta API calls)" to run it. After a run, the page shows the
    cached numbers at once. The card shows the same-direction
    count ("Same direction (up, down or flat, 0.1 margin): N of
    30") and the correlation of the two judges. The local judge
    gates the Create verdict. Muse Spark does not gate it.
11. **Tune with Muse**: Muse Spark picks up to 5 doses for the
    patch. Before each try, Muse writes a prediction: the expected
    gain and the guard call. The panel shows each prediction next
    to the measured result, the guard state, and the final verdict.
    Each try calls its guard check the "first-token check". The
    final check is the "full guard check". The local judge and
    the guard decide the verdict. The guard stays on.
12. **In plain words**: Muse Spark explains the report from its own
    numbers. With `META_API_KEY` in `.env`, it writes this as the page
    opens (one call per report; the server keeps it). Without the key,
    the section says what is missing.

## New behavior

Write the behavior in plain words, then pick one way:

- **Let Muse build it**: Muse Spark drafts the contrast prompts and the
  judge question, the model writes 20 replies, Muse labels them, and the
  local judge must agree on at least 85% before Create runs. If a gate
  fails, Muse may rewrite the description, up to 3 rounds. The run page
  shows every step as it happens and ends with "Open evidence" when a
  patch is made. Past runs are listed under "Muse runs".
- **Label it yourself**: the loaded model drafts the parts and writes 20
  replies. Mark each one "Shows it" or "Does not". "Ask Muse for
  suggestions" proposes labels; nothing is saved until you accept them.
  "Check the judge" registers the family when the judge agrees with you
  on at least 85%; then "Create it now" starts Create.

Below these, "Or start from a validated family" starts Create from a
family that already has a validated judge. Create takes about 8 to 13
min on the llama lane (`LOBOT_CREATE_LANE=llama`), about 35 to 50 min on
torch.

## Chat

The chat keeps the sentence's edits on for every turn. With "+ read",
a pen line under each reply reads that behavior word by word (the band
is 0 for neutral and 1 for the behavior's mean). The first turn waits
for a full-generation guard check, which can take a few minutes; the
result is kept for the same edits. Reading the behavior you push shows a
note: the reading then includes the push itself.

### Learned mixes in the app

A job with a passed learned confirm (`learned.json`) shows
`LEARNED PASS` in the shelf and the workbench list. "Use in the
sentence" loads the learned rows. The dose is a multiplier of the
tested mix: dose 1.0 is the tested mix. Lower and higher doses scale
the mix, but only the tested dose has a pass. A mix the full guard
blocks shows `GUARD` and runs only in Research mode. The evidence
page links the `learned.lobot` patch file (`GET
/api/traits/{job_id}/learned-patch`). The card links the job detail.

## Bring a command-line run into the app

`run_trait.py` writes to any folder. To see that run on the shelf:

```sh
uv run python scripts/import_run.py logs/<run> --id <name>
```

## Where each screen's code lives

- Workbench: `web/js/screens/workbench.js`, run flow in `run.js`.
- Evidence: `web/js/screens/evidence.js` and `pages.js`.
- Library: `web/js/screens/library.js`.
- New behavior: `web/js/screens/new-behavior.js`.
- Agent: `web/js/screens/agent.js`.
- Labeling: `web/js/screens/labeling.js`.
- Chat: `web/js/screens/chat.js`.
- Shared: `web/js/state.js`, `web/js/api.js`, `web/js/components/`
  (dial, chart, legend, strip), styles in `web/css/app.css`.
