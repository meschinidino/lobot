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
  split at the first changed word.
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
2. **The dose rule**: which rule chose the dose. A dose chosen before
   the fluency check shows a warning with its fluency.
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
10. **In plain words**: Muse Spark explains the report from its own
    numbers (needs `META_API_KEY` on the server).

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

## Bring a command-line run into the app

`run_trait.py` writes to any folder. To see that run on the shelf:

```sh
uv run python scripts/import_run.py logs/<run> --id <name>
```
