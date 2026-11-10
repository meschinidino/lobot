# Wednesday demo: five behaviors

Warning first: use only the job in each row. Other jobs for
the same trait scored near zero on these fresh prompts.

Model: `meta-llama/Llama-3.2-3B-Instruct`. Every behavior
below ran on real weights on 2026-11-09, plain vs steered
on the same five fresh prompts. Full replies:
`logs/demo5_measure.json` and `logs/demo5_measure2.json`.

## The slate

| # | Behavior | Job | Frozen | Hero prompt |
|---|---|---|---|---|
| 1 | Pirate | `pirate-fv2-l16` | L16 @ 1.0 | surprise birthday party |
| 2 | Old West cowboy | `cowboy-fv2-l20` | L20 @ 1.0 | learn to swim as an adult |
| 3 | Devout Catholic | `answers-like-a-devout-catholic-speaks-ab-agent-20261106-003239-2c12aa` | L10 @ 1.0 | dinner with eggs, spinach, and cheese |
| 4 | Shakespeare | `shakespeare-fv3` | L13 @ 0.5 | bicycle chain keeps slipping |
| 5 | Excited sports commentator | `sports-fv2-l20` | L17 @ 0.5 | ask my boss for a raise |

The five prompts, shared across every behavior in the
measure:

1. How do I fix a bicycle chain that keeps slipping?
2. What should I cook for dinner with eggs, spinach, and cheese?
3. How do I ask my boss for a raise?
4. What is the best way to learn to swim as an adult?
5. How do I plan a surprise birthday party for my sister?

Each hero prompt in the slate uses one of these five. On
stage: ask the hero prompt with the dial off, then turn
the dial on and ask it again.

## Live rehearsal (2026-11-09, through the server)

The five traits ran through `POST /api/update` on the
live server (`logs/demo_rehearsal.json`). The server
resolves each trait to its newest knob job, which is not
always the measured job. Stage doses:

| Behavior | Server serves | Stage dose | Note |
|---|---|---|---|
| Pirate | L13 direction (`metrics-pirate-c`) | 1.0 | flat at 0.5; voice shows at 1.0 |
| Cowboy | L20 (`cowboy-fv2-l20`) | 1.0 | hero text reproduced |
| Devout Catholic | L10 (agent job) | 1.0 | hero text reproduced |
| Shakespeare | L13 (`shakespeare-fv3`) | 0.5, no higher | 1.0 degrades into repetition of phrases |
| Sports commentator | L17 (`sports-fv2-l20`) | 0.5, no higher | 1.0 collapses into a `FOLAND` token loop |

The measured fv2 pirate job is stronger on paper, but
the shelf serves the L13 direction. At dial 1.0 the L13
reply opens "Me dear sister, ye'll be wantin' t'get the
ol' winkie wink, me boyo!" and reads as pirate. Use the
shelf as rehearsed; do not swap jobs on stage.

## Scores on the five prompts (mean trait gain)

| Behavior | Gain | Min | Fluency | Verdict |
|---|---|---|---|---|
| Pirate fv2 | 0.953 | 0.828 | 0.708 | stage-ready |
| Cowboy | 0.843 | 0.653 | 0.699 | stage-ready |
| Devout Catholic | 0.800 | 0.392 | 0.594 | stage-ready |
| Shakespeare fv3 | 0.263 | -0.027 | 0.989 | one hero prompt only |
| Sports commentator | 0.352 | 0.113 | 0.990 | stage-ready |

Fluency here is the local judge score. Pirate and cowboy
texts read clean to a human; the judge scores strong
dialect down. No slate reply collapsed (repetition check).

## Cut, with numbers

- Pirate `metrics-pirate-c` L13 @ 0.5: gain 0.015, weak
  on 5 of 5. The bank-final star does not transfer to
  these fresh prompts. Do not use it on stage.
- Noir detective L20 @ 1.0: gain 0.000. Dead.
- Pirate `pirate-cli-l17` L17 @ 1.5: replies collapse
  into repeated tokens on 3 of 5.
- Shakespeare `shakespeare-fv2-l20` L20 @ 1.0: gain
  0.793, but 2 of 5 replies loop words. Too risky live;
  fv3 is weaker and clean, so fv3 takes the slot with
  the bicycle prompt (gain 0.88, fluency 0.977).

## Spanish attempts (measured, do not use)

The same five jobs ran on Spanish translations of the
five prompts on 2026-11-09 (`logs/demo5_measure_es.json`).
The English-derived vectors do not transfer:

| Behavior | Gain in Spanish | Fluency | Reading |
|---|---|---|---|
| Pirate fv2 | 0.038 | 0.007 | broken text, token loops |
| Cowboy | 0.011 | 0.290 | broken Spanish, no voice |
| Devout Catholic | 0.045 | 0.058 | gibberish with laughter markers |
| Shakespeare fv3 | 0.001 | 0.887 | clean Spanish, no effect |
| Sports commentator | 0.296 | 0.864 | mild enthusiasm transfers |

Keep the stage prompts in English. A Spanish demo needs
Spanish-derived vectors, which is a new Create per
behavior, not a translation of the prompts.

The first ES derivation spike also failed (2026-11-09,
`logs/pirate_es_spike_v2`): a pirate direction derived
from Spanish contrast captures froze L14 @ 0.5 and scored
gain 0.027 on the five demo prompts. The steered replies
turn friendly, not pirate. The base model does produce
pirate Spanish when prompted (judge 0.23-0.55 vs 0.001
plain), so the style exists in the model. The
prompt-capture direction does not isolate it in Spanish.
Next ES attempt should derive from generated pirate
answers, not from instruction prompts.

## Fallback

If the live run misbehaves on stage, the measured plain
and steered replies for every row sit in
`logs/demo5_measure.json` (Catholic, cowboy, Shakespeare
fv3, sports, and the cut pirate and noir rows) and
`logs/demo5_measure2.json` (pirate fv2 and the other
alternates). Quote the hero-prompt row for the behavior.
