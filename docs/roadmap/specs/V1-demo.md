# V1 Recordings and demo production

Owner: human (recording, voice), model slot (recordings), Claude (screens,
share card, script). Depends on: R3, S1, U1-U3. Time: Sunday and Monday (final video Mon 11-03 23:00 ET).

## Goal

A recorded 3-minute video and an X post that show the loop on Llama, honest
and reproducible. Every run on screen plays from a real recording in replay
mode, so the take cannot fail. A live take is optional.

## Storyboard (3:00)

| Time | Beat | On screen | Voice-over (draft) |
|---|---|---|---|
| 0:00-0:20 | Hook | Collapsed shelf: the sentence and one answer | "This is Llama 3.2, on my laptop. I want to see inside it, change one thing, and prove the change is safe." |
| 0:20-0:55 | Read | Click a word; top neurons; open the card: fires on, pushes toward | "I click a word. These neurons fired here. This one fires on X and pushes toward X." |
| 0:55-1:35 | Steer | Push it; Run; the answer forks; proof line with the lift | "I push it. The answer forks right here. The reader predicted X; steering made X N times more likely." |
| 1:35-2:10 | Guard | Push a refusal-weakening edit; BLOCKED with the reason; lower it; it passes | "Now a bad edit. The guard checks it first: refusals fell under half. Blocked. Lower the dose, and it passes." |
| 2:10-2:40 | Prove and share | Evidence card; Download patch; llama.cpp flags | "Every change ships with its evidence and a safety record. Anyone can install it and re-check it." |
| 2:40-3:00 | Close | Share card | "See it, change it, prove it, keep it safe. On your own computer." |

## Steps

1. Model slot: record every demo setup with `record_sweep.py` (the R3 neuron,
   the guard edit at a blocked and a passing dose) so replay mode can play the
   whole video.
2. Claude: share card for the R3 neuron; X post text under 280 characters;
   final screenshots; collapsed-shelf recording view.
3. Human: record the replay take (the main take); record a live take only if
   time allows; cut to 3:00; captions on.
4. Rehearse twice from a cold start, timing each beat.

## Checks

- Every number on screen comes from a real run; the video says nothing the
  reports do not support.
- No harmful text on screen (the guard shows `display_prompt`).
- No event or competition reference in repo files.
- Replay output matches the recorded run token for token (replay tests pass).

## Done means

Video file, X post text, share card image, both takes, a rehearsal log.
