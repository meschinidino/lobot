# Morning backend report (model-lane night, Llama 3.2 3B)

All model work ran on meta-llama/Llama-3.2-3B-Instruct,
revision 0cb88a4f764b, one job at a time. Bookings in
`docs/roadmap/model-lane.md`.

## What merged into claude/admiring-mendel-mn88n8

- R2 reader API (lane/R2-reader-api): neuron card, find,
  lift, neurons reply. 287 tests pass.
- R3 validation (lane/R3-validation): fires_on cache
  (105,029 tokens, 151 s), search over layers 12-27
  (46 min, peak 6128 MiB), 378 scored, 57 passing,
  cards.json with 10 cards.
- D1 read dials (by the D1 lane, merged by the integrator)
  plus the Llama model check: baseline mean 0.527,
  steered mean 16.564.
- E1 evidence API (merged by the integrator, 78e4f52).
- S1 guard code (lane/S1-guard-code) plus held-out
  measurement: 0 misses, 1 false block, 0.37 s cold,
  instant warm.
- P1 patch code (lane/P1-patch-code) plus round trip:
  export ok, verify --quick PASS (shipped 0.771
  [0.728, 0.812], measured 0.738).
- Night data: 2 Llama sweeps, D1 numbers, demo-script
  values, lane bookings. Full suite: 324 pass, 5 skip.
- J1 skipped: META_API_KEY is not set.

## Demo values (also in demo-script.md)

- Prompt: "Tell me about your weekend."
- Word: "storm". Neuron: L19N4330.
- Fork dose: 1.0 ("I'm a stormy weather system...").
- Guard: prefill passes 1.0 (0.787), blocks 2.0 (0.026).
- Lift line: "storm" 45,709x at +1.0 (R3 words), 17,550x
  live in the UI (first up word).
- Behavior: pirate (Create 49 min, L17 @ 1.5, knob).

## What failed or needs a decision

1. Verified guard blocks the steer at every dose (1.0,
   0.5, 0.25) on a borderline prompt. Beats 2 and 3b need
   research mode, an S1 follow-up on borderline flips, or
   guard beats on L11N1343 (0.0 passes, 2.0 blocks).
2. Pirate evidence rows are red except Read (Increase
   FAIL on the control, Decrease -0.10, Delete -0.01).
   The Prove beat needs a passing trait or a new beat.
3. Pirate knob loops at dose 1.5 on the weekend prompt
   ("me me me..."). Lower the fan doses.
4. D11 gate prints FAIL: the held-out set has no true
   weakening edit (vacuous). Next run must guarantee one.
5. No llama.cpp or Ollama on this machine: GGUF export
   ok, install gate unrunnable, Ollama target untested.
6. UI findings B1-B6 and R1 in model-pass.md (web/ is
   Claude's; neuron cards need a replay path).

## What is left

- V1: lock the demo design around the guard findings,
  record the video.
- Optional: another family Create for green evidence
  rows (~50 min each); S1 re-split with a true positive;
  a llama.cpp machine for the install gate.
