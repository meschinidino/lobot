# PP Prompt Pareto

Owner: backend agent. Lane: no-model for code and tests; one model
slot for the live runs. Depends on: Create (`trait_eval.run_trait`),
the Create lanes. Time: one day.

## Goal

Every Create report answers one question with numbers: "Is this
patch better than just asking the model in the prompt?"

Published benchmarks (AxBench, SteerScope) find that a prompt usually
beats activation steering on effect, and that no steering method gets
more effect without more side effects. lobot measures both axes for
each patch, next to its own prompt baseline.

## Points

All points use the final split and the frozen layer. The frozen dose
stays as tuned; the Pareto is descriptive and chooses nothing.

1. `vector@d` for d in `{0.5, 1.0, 1.5, 2.0, 2.5}` plus the frozen dose.
2. `prompt`: the top baseline instruction (from the O4 baseline).
   `prompt:label`: "Answer the question. Your reply must match this
   description: {judge label}". Some drafted baselines are sample
   lines, not instructions (cowboy: "Howdy ma'am, ..."), so the top
   baseline alone can be a weak prompt.
3. `prompt+vector`: the top instruction plus the vector at the frozen
   dose.

## Metrics (per point, per prompt)

- `trait`: judge score for the family label, minus the base reply's
  score.
- `relevance`: judge score for "relevant to the question" (chosen by a
  label check on real weights; see the PP report).
- `fluency`: judge score for "written in fluent, coherent language".
- `cost`: mean of the relevance drop and the fluency drop against the
  base reply. A negative cost is an improvement.
- `hm`: harmonic mean of the three raw scores per reply (the AxBench
  aggregate shape), averaged over prompts.

## Verdict (rule fixed before any measurement)

- The verdict compares against the stronger prompt point: the one of
  `prompt` and `prompt:label` with the higher mean trait gain.
- A vector point beats the prompt when its mean trait gain is at least
  the prompt's and its mean cost is at most the prompt's.
- `vector_dominates`: at least one vector point beats the prompt.
- `prompt_dominates`: the prompt beats every vector point (its trait
  is at least theirs and its cost at most theirs).
- `tradeoff`: neither.
- The report also gives the best vector point by `hm`, with bootstrap
  95% intervals on its paired trait and cost differences from the
  prompt.

## Deliverables

- `src/model_lobotomy/pareto.py`: pure helpers (point summary, front,
  verdict) and `run_pareto(...)` on a Create lane.
- `run_trait(..., pareto=True)` writes `out["pareto"]`;
  `scripts/run_trait.py --pareto`.
- `scripts/pareto.py JOBDIR`: the same stage for an existing Create
  run (reads `report.json` and `direction.safetensors`), writes
  `pareto.json`.
- `tests/test_pareto.py` (fakes, no weights).
- Report `docs/roadmap/reports/PP.md`.

## Acceptance

1. Unit suite passes; the verdict rule has tests for all three
   outcomes.
2. Live on Llama 3.2 3B for pirate and cowboy: `pareto.json` written,
   verdict reported as measured, whatever it is.
