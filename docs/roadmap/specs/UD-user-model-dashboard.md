# UD "What it thinks of you" dashboard

Owner: Muse (coding agent). Lane: no-model for code and tests; one
model slot for probe training and the causal check; UI lane for the
panel. Depends on: CH (chat with dials), Read. Time: three days.

## Warning

The panel shows guesses the model makes about the person who chats
with it. These guesses can be wrong and can be sensitive. Rules:
- Show every value as "the model's guess", never as a fact.
- Keep everything local. Never send a reading to any service.
- Never store readings unless the user saves the transcript.
- No export of readings in patches or the registry.

## Goal

Show, live in chat, what the model assumes about the user, and let
the user pin or clear each assumption.

Basis: TalkTuner (Harvard; Chen, Viégas, Wattenberg). Linear probes on
the residual stream read a chatbot's internal model of the user (age,
gender, education, socioeconomic status). Steering along the probe
direction changes the replies.

## Attributes (fixed now)

| Attribute | Classes |
|---|---|
| age | child, adolescent, adult, older adult |
| gender | female, male |
| education | some schooling, high school, college and more |
| socioeconomic status | low, middle, high |

## Data

1. Muse writes synthetic conversations: for each attribute class, 150
   conversations of 2-4 user turns. The user persona is given in the
   generation prompt only. The conversation text never names the
   attribute directly ("I am 9" is not allowed; topic and style may
   suggest it). A test rejects any conversation that contains a list
   of banned direct words per class.
2. Split per class: 100 train, 25 validation, 25 test. Commit the text
   in `store/user_model/` (it is synthetic; no real user data).

## Probes

1. Capture the residual at the last token of the last user turn, all
   layers (torch, cached).
2. One logistic-regression probe per attribute and layer. Choose the
   layer on validation. Report test accuracy once.
3. Control: the same probe on shuffled labels.

## Pin and clear

- **Pin** class c: add `alpha x unit(w_c)` at the chosen layer on
  every token (a vector edit, same as patches).
- **Clear**: project the attribute's probe span out of the residual
  at that layer (an ablation edit).
- Causal check: 20 held-out prompts with no attribute cue. For "pin
  child", the judge label "The reply is written for a child". Paired
  gain vs base and vs a random direction of equal norm.

## UI

- A panel in chat: one row per attribute, bars per class from the
  probe, updated after each user turn. Each row has Pin (pick a class)
  and Clear.
- One line at the top: "These are the model's guesses about you. They
  stay on this computer."
- Follow docs/DESIGN.md. Ask the user before any new visual pattern.

## Routes

- `GET /api/chat/{session_id}/user-model`: latest readings.
- `POST /api/chat/{session_id}/user-model` with
  `{"attribute": "...", "action": "pin" | "clear" | "off", "class": "..."}`.
- Document both in `docs/reference/api.md`.

## Files

- `src/model_lobotomy/user_model.py`: probes (fit, read), pin and
  clear edits.
- `scripts/user_model.py fit|check`.
- Tests: probe fit on fake activations, banned-word filter, edit
  shapes, routes with a fake engine.

## Acceptance (fixed now)

1. Unit suite passes.
2. Probe test accuracy >= 0.80 for at least 2 of 4 attributes, with
   the shuffled control at chance +- 0.10.
3. Causal: "pin child" and "pin older adult" each give a paired gain
   with a CI above 0 and above the random-direction control.
4. Clear: after clear, the probe reading for that attribute is at
   chance on the test conversations, and fluency stays >= 0.5.
5. Panel works in replay mode; screenshot in the report.

## Known risk

Synthetic personas can teach the probe the writer's stereotypes, not
the model's. Report test accuracy on conversations that a second
writer prompt produced (a different Muse instruction), as a transfer
check.
