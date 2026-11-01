# TD Trained directions

Owner: Muse (coding agent). Lane: no-model for code and tests; model
slot for training and the live runs. Depends on: FD or JS, PP. Time:
three days.

## Goal

Train the steering vector instead of taking a mean difference. Aim:
a patch that is not dominated by its own prompt, and a working
sycophancy patch (DiffMean failed on it).

Basis:
- RePS (Stanford): a two-way preference objective. The vector at a
  positive factor must prefer the steered reply; at a negative factor
  it must prefer the plain reply.
- "Steer like the LLM" / prompt steering replacement (2026): distill
  what the system prompt does into the activation edit.

## Step 0: a sycophancy family

The store has no sycophancy family; the spike used an ad-hoc one. Add
`agrees-with-the-user-even-when-wrong` through the C3 open-description
path (Muse draft, judge label validated on hand labels, agreement >=
0.8). If the judge cannot see sycophancy (agreement < 0.8), drop Gate
A and say so in the report.

## Data

For a family, on its extraction and tuning prompts only:
1. Positive reply: the model's own greedy reply with the family's
   positive system prompt (`cfg["pos"]`).
2. Plain reply: the greedy base reply.
Use 60 pairs for training, 20 for validation (both from extraction and
tuning). The final split stays untouched.

## Objective

Trainable: one vector v (3072) at the chosen layer. The model stays
frozen (`requires_grad` off on all weights, as GR does).
- Up loss: -log sigmoid(beta x (log p(pos | +v) - log p(plain | +v))).
- Down loss: -log sigmoid(beta x (log p(plain | -v) - log p(pos | -v))).
- Total: up + down + lambda x ||v||^2. beta = 1, lambda = 1e-3.
- The prompt is the plain user turn (no system prompt) in both terms.
- Adam, lr 1e-2, 200 steps, batch 4, sequence cap 128 tokens.
- Initialize v from the DiffMean direction.

GR already ran backward passes through the 3B model on the Mac, so
memory should fit. Measure peak memory on step 1 and log it.

## Dose

The trained v has its own norm. Sweep factors {0.25, 0.5, 1, 1.5} of
that norm on tuning with the current dose rule.

## Files

- `src/model_lobotomy/train_direction.py`: data, loss (pure parts
  tested), train loop.
- `run_trait(..., direction="trained")` and
  `scripts/run_trait.py --direction {diffmean,trained}`.
- Tests: loss sign on toy logits, frozen weights stay unchanged, the
  vector is the only parameter with a gradient.

## Acceptance (fixed now)

Live on Llama 3.2 3B: pirate, cowboy, sycophancy, noir.
1. Unit suite passes.
2. Gate A: sycophancy's trained patch passes the turn-up card.
3. Gate B: for at least one family, the PP verdict is not
   `prompt_dominates`, and the trained patch's Pareto score (hm) at
   its dose is >= the DiffMean patch's hm + 0.05.
4. Report all four, both directions, pass or fail.

## Known risk

The positive replies come from the prompt, so the vector learns to
imitate the prompt. It can match the prompt at best on trait, but it
may cost less. That is the result to look for.
