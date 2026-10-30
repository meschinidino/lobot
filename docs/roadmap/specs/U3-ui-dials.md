# U3 UI: read dials on the tree

Owner: Claude (UI lead). Lane: UI. Depends on: contracts section 3 (D1).
Time: half a day.

## Goal

"See it" without a dashboard: for a saved behavior, show how strongly it is
present at each word of each run, inside the tree, not in a panel of meters.

## Design

- In the sentence: "and read [behavior ▾]" adds a read (max 2).
- Each branch's words carry a thin pen line under them (the channel strip from
  DESIGN.md, Read detail): one step per token, 0..1 band shaded. One pen color
  per behavior, from the pen inks; always with a text label.
- The selected run's pen draws as the run completes (420 ms), not before.
- The word inspector shows the exact value at the clicked word:
  "hostility 0.92 (observed projection; 1 = trait mean)".
- Pink is not used for dials. The onset mark (first crossing of 1.0) is ink.

## Checks

- Against D1 stubs, then the real API. Values only from the backend.
- Lint 0/0; 375/768/1440 px; reduced motion shows the final state;
  screenshots to the user.

## Out of scope

Streaming dials per token while generating (later).
