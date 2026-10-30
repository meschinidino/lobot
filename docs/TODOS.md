# TODOs

The ordered plan is `docs/roadmap/` (10X specs) and `docs/roadmap/later.md` (100X, 1000X). This file keeps deferred items with their reasons.

## Bake Delete into model weights and export a full model (deferred 2026-10-29)

Run-time Delete (directional ablation) works only inside lobot. Baking it in means orthogonalizing the weights against the trait direction and exporting a full model file (several GB, for example GGUF for llama.cpp).

- Why deferred: XL effort (weight edit plus conversion pipeline), multi-GB outputs, and it is the direct path to distributing models with safety refusals removed.
- Depends on: run-time Delete with report cards, the `.lobot` patch format, and the refusal guard.
- Must keep: the refusal guard and the guard state recorded in every exported artifact.

## Open-ended Create for descriptions outside the validated families (deferred 2026-10-29)

- What: Create accepts any description. The model writes the contrast instructions and the judge question; about 20 user labels per new trait validate the judge before a report card counts.
- Why: wave 1 limits Create to validated trait families (eng review D12). The open "type any trait" flow needs this pipeline.
- Pros: restores the open flow with per-trait validation.
- Cons: a 3B model may write weak contrasts; Create gains a labeling step.
- Context: `scripts/spike_knob.py` BEHAVIORS shows the hand-written parts each trait needs today. Start with a bounded spike: 2 new traits, model-written parts, 20 labels each, about 1 hour.
- Effort: human L / CC M. Priority: P2.
- Depends on: wave-1 trait families, the context-aware judge validated on hand labels, and the final-split policy (one final check per frozen candidate).

## Make the model choice persistent and default to Llama (2026-10-30)

- What: the hero model is Llama 3.2 3B, but `steer.MODEL_ID` (Qwen) is still the startup default. Remember the last model picked in the shelf, and start on Llama when its weights are present.
- Why: the demo and the roadmap run on Llama; a cold start on Qwen costs a model swap (about a minute) and RAM.
- Owner: U1 (UI) with a small server change. Priority: P1.
