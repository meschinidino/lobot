# U2 UI: reader in the word inspector, find-a-neuron

Owner: Claude (UI lead). Lane: UI. Depends on: contracts section 2 (R2).
Time: one day.

## Goal

Beat 1 and the proof for beat 2, on screen: click a word, see the neurons
behind it and what they mean, push one, and see the lift.

## Design

- **Word inspector** (already has the slot): "Top neurons at this word" lists
  `POST /api/neurons` results (with `reply`): neuron id (mono), activation and z, and the
  top 3 "pushes toward" words. Each row: Push (adds the neuron to the sentence
  at +1.00) and Open (neuron card).
- **Neuron card** (side panel, not a page): two labelled lines, "Fires on"
  (3 contexts with the token marked, from `fires_on`) and "Pushes toward"
  (up words; down words folded). Show "Pushes toward" only when coherence >=
  0.5; otherwise "Not readable at this layer" with the coherence value. A
  one-line note: "Neurons often mix several jobs."
- **Proof line** after a push and run: "The reader predicted *X*; steering
  made *X* N× more likely" from `POST /api/lift`. Shown only with a real lift
  value; never estimated.
- **Find a neuron**: a field in the shelf, "Find neurons that push…", calls
  `GET /api/find`, lists results; each opens the card.
- Pink stays on the split only. The marked token in "Fires on" uses an ink
  underline, not pink.

## Checks

- Works against R2 stubs first, then the real API; replay mode serves cards
  from `cards.json`.
- Playwright flows: click word → list → push → run → proof line; find → card.
- Lint 0 blocks, 0 asks; 375/768/1440 px; keyboard; screenshots to the user.

## Out of scope

The reader math (R1), the search for demo neurons (R3).
