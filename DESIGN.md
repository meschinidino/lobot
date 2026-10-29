---
# gstack: design-md-format=spec
name: lobot
description: A clinical ward chart that stays orderly until one token, where a hot pink cut shows that one knob changed a mind.
colors:
  ground: "#D5E3D0"
  sheet: "#EEF4EB"
  grid: "#B7CCB2"
  text: "#1E2924"
  text-muted: "#41504A"
  primary: "#7B1D2B"
  primary-hover: "#651522"
  on-primary: "#FBF1EF"
  cut: "#FF4F9A"
  on-cut: "#1E2924"
  success: "#2F6B4A"
  success-tint: "#D3E6D8"
  warning: "#8A5A00"
  warning-tint: "#F3E6C4"
  error: "#B3261E"
  error-tint: "#F6DAD7"
  dark-ground: "#131C18"
  dark-sheet: "#1C2722"
  dark-grid: "#24332C"
  dark-text: "#DCE7DF"
  dark-text-muted: "#9DB0A6"
  dark-primary: "#B0404F"
  dark-primary-hover: "#9A3240"
  dark-cut: "#FF5FA4"
  dark-on-cut: "#131C18"
  dark-success: "#8FD3A8"
  dark-warning: "#F0C46A"
  dark-error: "#FF8A80"
typography:
  display:
    fontFamily: Bricolage Grotesque
    fontWeight: 800
    fontStretch: 75%
    fontSize: clamp(48px, 7vw, 96px)
    lineHeight: 0.9
    letterSpacing: -0.02em
  heading:
    fontFamily: Bricolage Grotesque
    fontWeight: 800
    fontStretch: 75%
    fontSize: 28px
    lineHeight: 1
  output:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 18px
    lineHeight: 1.6
  body:
    fontFamily: Atkinson Hyperlegible Next
    fontSize: 1rem
    lineHeight: 1.5
  label:
    fontFamily: Atkinson Hyperlegible Next
    fontWeight: 700
    fontSize: 14px
  mono:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontFeature: tnum
rounded:
  none: 0px
  sm: 2px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
  3xl: 72px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.none}"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
  button-ghost:
    backgroundColor: transparent
    textColor: "{colors.text}"
    borderColor: "{colors.text}"
    rounded: "{rounded.none}"
  input:
    borderColor: "{colors.text}"
    rounded: "{rounded.none}"
  sheet:
    backgroundColor: "{colors.sheet}"
    borderColor: "{colors.text}"
    rounded: "{rounded.none}"
  split-token:
    backgroundColor: "{colors.cut}"
    textColor: "{colors.on-cut}"
---

# lobot

## Overview

**Creative North Star:** A ward chart that goes wrong. The page is calm clinical paperwork. One hot pink cut marks the token where the steered answer leaves the baseline.

**Memorable thing:** "I watched one knob change a mind." Every design choice serves the before/after split.

**Product context:** lobot is a local neuron-steering workbench. It runs one prompt twice on a small open model (baseline and steered) and shows the first divergent token. Users: portfolio viewers who look once (often as an X screenshot), and the author, who runs many experiments. Peers: Neuronpedia, Goodfire, Transluce. All three use calm, light, academic looks. lobot breaks from them on purpose.

**Mode per surface:**
- Workbench: Operate. Dense, fast, keyboard-first.
- First visit: Persuade. A recorded example run explains the product before weights load.
- Capture view: Experience. A portrait card built for a phone feed.

**Reference sites:** https://www.neuronpedia.org, https://www.goodfire.ai, https://transluce.org (screenshots taken 2026-10-29).

**Key characteristics:**
- Ward-green ground with a faint chart grid.
- Big condensed display type, used once per screen.
- Square corners, ruled fields, 2px ink borders, no shadows.
- One pink mark per run: the split.

## Colors

**Strategy:** Committed. Ward green owns the page. Neutrals are green-tinted inks.

**Light or dark:** Light ("Day ward") is the default. The use scene is a laptop in mixed light, and screenshots go to X feeds that are often light. Dark ("Night ward") is a full token set, not an inversion. It keeps the same hierarchy: sheet is one step lighter than ground, and pink stays the brightest color on the page.

Rules for each token:
- `cut` (pink) marks the first divergent token, the cut line and the spike in the agreement trace. Use it nowhere else, not even for focus rings.
- `primary` (oxblood) is the run action only.
- `success`, `warning` and `error` signal run states with a tint background and a colored mono label. Do not use colored edges.
- `text-muted` is for metadata. Never use it to fade the baseline answer: both answers must stay fully readable.

## Typography

All three faces load from Google Fonts. Self-host them in the app, because lobot runs local only.

- **Bricolage Grotesque** (800, width 75): wordmark, page headline, section heads and the BASELINE / STEERED column heads. Never body text.
- **Atkinson Hyperlegible Next:** model output (18px/1.6), UI (16px) and labels (14px bold, sentence case, no tracking). It is a legibility face, which fits the clinic theme and survives phone-size screenshots.
- **JetBrains Mono** (tabular numbers): layer, neuron, dose, token index, timings and the recipe string. Every value a user might copy is mono.

Scale: display 48 to 96px, heading 28px, output 18px, body 16px, label 14px, mono 12 to 14px.

Capitals appear only in the display voice (BASELINE, STEERED) and in mono state keys (RUNNING, COMPLETE). Labels and table headers use sentence case. Levels differ by size, not only by weight.

## Layout

- Max width 1400px. Outer margin 32px on desktop, 16px below 720px.
- Masthead: wordmark left; model picker ("PATIENT: <model>"), History and theme toggle right.
- First visit: a headline band ("One knob. Different answer.") over a recorded example run. After the first real run, the band collapses into the masthead.
- Workbench: a 300px left rail (prompt, recipe table, add row, Run experiment) and one shared output sheet. At laptop widths the rail can shrink to 260px. Below 1000px the rail moves above the sheet. Below 720px the two answers stack.
- Output sheet, top to bottom: run-state bar, 72px agreement trace, BASELINE | STEERED columns with equal width and aligned starts, export rows (recipe string, X post).
- A Workbench toggle compacts spacing for long sessions.

## Elevation & Depth

There are no shadows and no glow. Depth comes from layers of paper: the ground carries the grid, and the sheet sits on top with a 2px ink border. The only displacement is the cut: the steered text below the split shifts 2px right and 6px down, with a torn edge.

## Shapes

Corners are square (0px). Use 2px only where a browser control needs it. No pills, no rounded cards, no nested cards. Rows in the recipe table are table rows, not cards.

## Components

- **Focus:** every control shows a 3px `text` (ink) outline with a 2px offset. Ink reaches 3:1 on ground, sheet and grid; pink does not.
- **Run experiment (primary):** oxblood fill, `on-primary` bold text (tinted paper, not pure white), `⌘ Enter` hint. Hover: `primary-hover`. Active: shifts 1px down. Disabled while a run is active; Stop takes its place.
- **Ghost button:** transparent, 2px ink border, 44px minimum height. Hover and `aria-pressed="true"` invert to ink fill. Disabled: 45% opacity, `not-allowed` cursor and the `disabled` attribute.
- **Field:** no box. A 2px ink underline. Invalid: `error` underline and text, `aria-invalid`, one message under the recipe table.
- **Recipe row:** mono layer, neuron and signed dose (always show the sign, two decimals). The dose field is text, not a number input, so the sign and the decimal point do not follow the browser locale. A slider below each row with a fixed zero notch. Keyboard entry and fine steps work. Moving a dose changes the recipe only; generation runs on Run.
- **Model picker:** shows readiness ("warm", "gated, needs login") and the fixed note "Switching reloads weights."
- **Run states:** RUNNING (with token count and elapsed time), COMPLETE ("Answers split at token N"), NO SPLIT ("Both answers match token for token"), ERROR (factual text, for example "Layer 41 does not exist"). If one answer ends first, mark the end position as the split.
- **Agreement trace:** flat while tokens match, a pink spike at the split, jittery after it. The jitter shows the state "diverged"; it does not plot per-token data. The label "first split · token N" sits next to the spike.
- **Split token:** pink background, ink text, bold. The matching baseline token gets a 3px pink underline.
- **Jump to split:** scrolls both columns to the split. Scrolling is linked by default, with an unlock control.
- **Export rows:** recipe string with Copy; X post with a 280 counter and Copy for X.
- **History:** a drawer. Each item shows the prompt excerpt, recipe, split position and time. Selecting an item restores its settings and outputs.
- **Empty state:** a dashed ink box with one instruction sentence. It points to the recorded example.
- **Capture view:** hides controls. Shows wordmark, model, recipe, prompt and the excerpt around the split at 24px. Portrait layout for phone feeds. Label excerpts as excerpts.

## Do's and Don'ts

- Do: keep exactly one pink element group per run (split token, cut line, trace spike).
- Do: show signed doses with two decimals in mono (`+1.00`, `−1.00`).
- Do: design loading, no-split, error and long-output states for every view.
- Do: use real recorded runs in demos and the first visit. Label illustrative text as illustrative.
- Don't: add shadows, glow, gradients or rounded cards.
- Don't: use pink for buttons, links or decoration.
- Don't: turn the steered answer after the split into a red diff wall. After the split, text is ordinary ink.
- Don't: add brains, scalpels, glitch effects or invented sinister output. The real divergence supplies the disturbance.

## Motion

- **Approach:** intentional.
- **Easing:** enter ease-out, exit ease-in, move ease-in-out.
- **Duration:** micro 50 to 100ms, short 150 to 250ms, medium 250 to 400ms, long 400 to 700ms.
- **The one authored moment:** the cut. When the steered pass reaches the split, the pink line draws across both columns from left to right in 420ms, then the text below shifts and tears. With `prefers-reduced-motion`, show the final state without animation.

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-29 | Initial design system created | Created by /design-consultation from the README, the office-hours design doc and screenshots of three interp sites. Three independent drafts (Claude, Codex, a Claude subagent) converged on a clinical document with one hot mark at the split. |
| 2026-10-29 | Tokens taken from the approved HTML preview | AI mockups were not generated: the gstack designer has no OpenAI API key. |
| 2026-10-29 | Focus rings in ink, not pink | /design-html lint: pink measured 2.3:1 on the ground; focus needs 3:1. This also keeps pink for the split only. |
| 2026-10-29 | Labels in sentence case, untracked, 14px | /design-html lint blocks tracked uppercase labels. Capitals stay in the display voice and in mono state keys. |
| 2026-10-29 | `on-primary` #FBF1EF, dark hover #9A3240 | Lint flags pure white as untinted paper; the old dark hover was 4.4:1 against white text. |
| 2026-10-29 | Workbench layout exceptions | The run fills the first screen (no tall hero), model output may exceed the word budget per screen, and the bench uses no marketing section gaps. Waived in `web/index.html` with reasons. |
