# Demo recording script (V1)

The 3-minute video, click by click. Storyboard and voice-over: `specs/V1-demo.md`.
Fill the `<...>` slots from the R3 report (demo neuron, its words) and the
recordings Muse makes with `scripts/record_sweep.py`.

## Before you record

1. Mac on power, other apps closed, Llama 3.2 3B active (shelf, Models).
2. Browser window 1440 × 900, zoom 100%, Dark theme, no extensions in view.
3. Clear the experiment for the demo prompt if it holds old runs: open the
   browser console and run `localStorage.clear()`, then reload.
4. Decide the take:
   - **Live take** (needs weights): `uv run python server.py`. Every beat works.
   - **Replay take** (no weights, cannot fail): `MODEL_LOBOTOMY_REPLAY=1 uv run python server.py`. Run with no edit plays the baseline of any recording of the prompt.
     Run, the guard block and the share card work from recordings. Clicking a
     word (neuron list) and the lift proof need the model, so record beat 1
     live, or show the neuron card from `store/reader/<slug>/cards.json`.
5. Do one dry run of every beat. Check that no number on screen comes from the stub.

## Beats

| Time | Beat | Do this | Check on screen |
|---|---|---|---|
| 0:00 | Hook | Collapse the shelf (◂). Remove all edits (×). Type `Tell me about your weekend.`. The sentence reads "On Llama-3.2-3B-Instruct, with no edit … and ask". Press Run. | One answer, run 1, NO EDIT. (Run with no edit sends one neuron at dose 0, which changes nothing, and keeps only run 1.) |
| 0:20 | Read | Click the word `storm` in run 2 (the steered answer; run 1 has no storm word). The inspector lists the top neurons. Click `L19N4330`. | Card: "Pushes toward" `storm, storms, hurricane`; "Fires on" 3 contexts with the token underlined. (Run 1 click alternative: `find_neurons("storm")` ranks L19N4330 first.) |
| 0:55 | Steer | Press Push in the card. Turn the knob to `1.0` (arrows: 0.05, Page Up: 0.5). Press Run. | The answer forks ("I'm a stormy weather system..."); pink split word and curve on the new run; status line adds "The reader predicted …; … 45,709× more likely (measured)". |
| 1:35 | Guard blocks | Remove the storm edit (×). Press `+ edit`, `Push a neuron`, layer `11`, neuron `1343`, dose `2.0`, Add. Press Run. | BLOCKED key; note names the edit and the flipped confirm prompt. (L19N4330 at 1.0 also blocks the verified guard — real refusal loss, see S1-followup.md — so the guard beats use the refusal preset L11N1343 instead.) |
| 1:50 | Guard passes | Turn back to `0.0`. Run. | CHECK PASSED; new branch. |
| 2:10 | Prove | Open the shelf. Click the behavior `pirate` row. | Evidence page: Read / Increase / Decrease / Delete rows, guard, judge. Point at Download patch. |
| 2:40 | Close | Back to the tree. Share card. | 4:5 card: fork, pink split, dose, proof line. |

Optional, if time allows: "+ read `pirate`" before a run to show the dial line (baseline mean 0.53, steered mean 16.56 at dose 1.5; note the steered text loops — lower the fan doses)
under the words; "Run at 5 doses" with the behavior in the sentence for the fan (turn the knob to 1.0 first).

## After recording

- Copy the post text from the share card (Copy for X). It holds only recorded
  text and measured numbers.
- Cut to 3:00, captions on. Final video by Mon 11-03 23:00 ET.
