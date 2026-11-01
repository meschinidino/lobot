# LQ Ask the activations (stretch)

Owner: Muse (coding agent). Lane: model slot for most of the work.
Depends on: store families, capture. Time: four days. Risk: high.

## Warning

This needs a LoRA fine-tune of a 3B model on a 24 GB Mac. It may not
fit or may train too slowly. Stop and report if step 1 of training
does not fit in memory, or if one epoch takes more than 3 hours.

## Goal

Answer plain-language questions about an activation. Example: given
the activations of a reply, "What persona is the model using?" gives
"a pirate".

Basis: LatentQA (Berkeley; Pan et al.). A decoder model, fine-tuned
with LoRA, reads activations patched into its early layers and answers
questions about them.

## Design

1. Decoder: the same Llama 3.2 3B weights with a LoRA adapter (rank
   16 on q, v projections). The base weights are shared with the
   target, so only one model is in memory.
2. Read pass: run the target (adapter off) on the stimulus; capture
   the residual at layer 15 for the last 32 tokens.
3. Decode pass: adapter on; the prompt has 32 placeholder tokens; the
   captured activations replace the residual at layer 2 at those
   positions; then the question; train on the answer tokens only.

## Data

From the store families: stimulus = a user prompt with the family's
positive system prompt. Questions, written now:
- "What persona or style is the model using?" -> family description.
- "Is the model trying to follow a special instruction?" -> yes / no
  (no for base stimuli).
- "Describe the tone of the reply." -> the judge label.
Hold out 3 families completely for the test.

## Files

- `src/model_lobotomy/latent_qa.py`: patching, dataset, train, ask.
- `scripts/latent_qa.py train|ask|eval`.
- Tests: the patch hook places activations at the right positions
  (tiny model), the loss masks the question.

## Acceptance (fixed now)

1. Unit suite passes.
2. On the 3 held-out families, the judge rates the answer to the
   persona question as correct for >= 60% of stimuli.
3. Control: the same questions with the activation patch off (zeros)
   are correct for <= 30%.

## Known risk

The decoder can learn to guess from the question alone. The zero-patch
control measures that.
