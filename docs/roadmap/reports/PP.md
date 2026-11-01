# PP report: prompt Pareto

Spec: [specs/PP-prompt-pareto.md](../specs/PP-prompt-pareto.md).

## Results first

- **Verdict on both live patches: `tradeoff`.** At every dose that moves
  the trait, a plain prompt gives more trait gain at a lower cost. The
  vector wins only at dose 0.5, where it costs almost nothing but gains
  little (+0.18 to +0.21).
- **The shipped pirate patch makes broken text.** Its frozen dose (1.5)
  has fluency 0.017 and relevance 0.022 ("Yeavey well yer ol' ol' ol'
  ol' ..."). The trait judge still scores it +0.757. Create's dose choice
  does not see this. The Pareto does.
- **Two drafted baselines are not instructions.** Cowboy's top baseline
  is "Howdy ma'am, what seems to be the problem?" and noir's is "Describe
  a peaceful forest." The new `prompt:label` point is built from the
  judge label, so the comparison is fair for open families.

## What changed

- `src/model_lobotomy/pareto.py`: points, verdict, front, paired
  bootstrap differences, `run_pareto` on a Create lane.
- `run_trait(..., pareto=True)`, `scripts/run_trait.py --pareto`,
  `scripts/pareto.py JOBDIR` (writes `pareto.json`).
- `trait_eval.open_lane`, `trait_eval.load_job` (shared by the new
  scripts; `run_trait.py` uses `open_lane`).
- `tests/test_pareto.py`: 8 tests (verdict rule, all three outcomes,
  front, strongest prompt, dose grid, row math, the stage on a fake
  lane).
- Full suite: 671 tests pass, 5 skipped (fakes only, no weights).

## Relevance label check (real weights)

The first live run used "a direct answer to the question". The base
replies scored only 0.09 on it, so relevance hardly counted in the cost.
I checked five labels, fixed before measuring, on 40 base replies
(Llama 3.2 3B): each reply under its own question vs. a shifted question.

| Label | Own | Other | Own > other |
|---|---|---|---|
| a direct answer to the question | 0.105 | 0.010 | 95.0% |
| an answer to this question | 0.714 | 0.129 | 97.5% |
| **relevant to the question** | **0.911** | **0.130** | **97.5%** |
| about the same topic as the question | 0.828 | 0.110 | 97.5% |
| a reply that addresses what the question asks | 0.364 | 0.036 | 97.5% |

Rule: best pairwise accuracy, then the largest gap. "relevant to the
question" wins; `PARETO_VERSION` is `pp-v2`. The v1 tables are kept in
`logs/pp_v1/` (git-ignored).

## Live results (pp-v2)

Llama 3.2 3B, llama lane (f16), final split n=30, 60 tokens.
Command: `uv run python scripts/pareto.py <JOBDIR> --lane llama`.
`*` marks the Pareto front.

Pirate (`logs/clane_pirate_llama`, L17, frozen dose 1.5). Base: trait
0.081, relevance 0.931, fluency 0.986.

| point | trait gain | cost | relevance | fluency | hm |
|---|---|---|---|---|---|
| vector@0.5 * | +0.207 | +0.018 | 0.906 | 0.976 | 0.377 |
| vector@1 | +0.855 | +0.667 | 0.202 | 0.382 | 0.289 |
| vector@1.5 | +0.757 | +0.939 | 0.022 | 0.017 | 0.024 |
| vector@2 | +0.667 | +0.936 | 0.027 | 0.019 | 0.030 |
| vector@2.5 | +0.678 | +0.925 | 0.035 | 0.031 | 0.041 |
| prompt | +0.909 | +0.226 | 0.589 | 0.877 | 0.757 |
| prompt:label * | +0.910 | +0.205 | 0.646 | 0.862 | 0.795 |
| prompt+vector | +0.807 | +0.935 | 0.021 | 0.027 | 0.028 |

Verdict `tradeoff` vs `prompt:label`. Best vector by hm (vector@0.5)
minus the prompt: trait −0.704 [−0.820, −0.572], cost −0.187
[−0.231, −0.143].

Cowboy (`logs/c3_cowboy_llama`, L17, frozen dose 0.5). Base: trait
0.129, relevance 0.931, fluency 0.986.

| point | trait gain | cost | relevance | fluency | hm |
|---|---|---|---|---|---|
| vector@0.5 * | +0.179 | +0.019 | 0.900 | 0.979 | 0.447 |
| vector@1 | +0.714 | +0.353 | 0.484 | 0.728 | 0.530 |
| vector@1.5 | +0.754 | +0.936 | 0.010 | 0.035 | 0.018 |
| vector@2 | +0.672 | +0.937 | 0.011 | 0.032 | 0.019 |
| vector@2.5 | +0.422 | +0.946 | 0.020 | 0.004 | 0.010 |
| prompt * | +0.743 | +0.049 | 0.860 | 0.959 | 0.862 |
| prompt:label * | +0.865 | +0.237 | 0.630 | 0.812 | 0.760 |
| prompt+vector * | +0.861 | +0.147 | 0.697 | 0.926 | 0.838 |

Verdict `tradeoff` vs `prompt:label`. Best vector by hm (vector@1)
minus the prompt: trait −0.151 [−0.250, −0.064], cost +0.115
[+0.004, +0.230].

## Acceptance

1. Unit suite passes; the verdict rule has tests for all three
   outcomes. **PASS.**
2. Live on Llama 3.2 3B for pirate and cowboy: `pareto.json` written,
   verdict reported as measured. **PASS** (both `tradeoff`).

## Known gaps

- The judge is the local 3B model. Fluency and relevance are judge
  scores, not human ratings.
- Relevance is low for every style point, prompts included (0.59-0.86).
  Some of that is the style itself, not lost content.
- Create still picks its dose from trait gain and NLL on tuning. The
  pirate result shows that this lets a degenerate dose through. Adding
  the fluency score to the dose choice is open (see TODOS).
