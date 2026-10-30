---
# gstack: design-md-format=spec
name: lobot
description: A workbench on dark instrument paper where every run forks the answer into a new branch, and one knob turns the edit.
colors:
  ground: "#D5E3D0"
  sheet: "#EEF4EB"
  grid: "#B7CCB2"
  band: "#C6D9C0"
  text: "#1E2924"
  text-muted: "#41504A"
  primary: "#7B1D2B"
  primary-hover: "#651522"
  on-primary: "#FBF1EF"
  cut: "#FF4F9A"
  on-cut: "#1E2924"
  pen-1: "#1E2924"
  pen-2: "#24557A"
  pen-3: "#5B5F8C"
  pen-4: "#6E4A2F"
  success: "#2F6B4A"
  success-tint: "#D3E6D8"
  warning: "#8A5A00"
  warning-tint: "#F3E6C4"
  error: "#B3261E"
  error-tint: "#F6DAD7"
  dark-ground: "#131C18"
  dark-sheet: "#1C2722"
  dark-grid: "#24332C"
  dark-band: "#2A3D33"
  dark-text: "#DCE7DF"
  dark-text-muted: "#9DB0A6"
  dark-primary: "#B0404F"
  dark-primary-hover: "#9A3240"
  dark-cut: "#FF5FA4"
  dark-on-cut: "#131C18"
  dark-pen-1: "#DCE7DF"
  dark-pen-2: "#8DB7DA"
  dark-pen-3: "#A9ADD6"
  dark-pen-4: "#C9A27E"
  dark-success: "#8FD3A8"
  dark-warning: "#F0C46A"
  dark-error: "#FF8A80"
typography:
  display:
    fontFamily: Bricolage Grotesque
    fontWeight: 800
    fontStretch: 75%
    fontSize: clamp(64px, 9vw, 96px)
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
    fontSize: 20px
    lineHeight: 1.4
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
  channel-band:
    backgroundColor: "{colors.band}"
  split-token:
    backgroundColor: "{colors.cut}"
    textColor: "{colors.on-cut}"
---

# lobot

## Overview

**Creative North Star:** The fork. The model writes one answer. At the token where an edit changes its mind, the answer forks, and every run adds a branch, so tinkering grows a tree. One knob turns the selected edit. Printed flat on dark instrument paper: ink and one pink cut, never light or glow.

**Memorable thing:** "I could see it, and I could turn it." Turn the knob, and you see exactly where and how the answer changes. Every design choice serves this line.

**Product context:** lobot is a local workbench to see a behavior inside a small open model, change it, prove the change, and keep it safe, on one laptop. The loop: Read (click a word: the neurons behind it, what they fire on, what they push toward), Steer (push a neuron or a behavior direction; the answer forks), Guard (the refusal check blocks unsafe edits and says why), Prove (report card and a patch others can re-check). Behavior CRUD (Create, Read, Update, Delete) is the engine underneath. Hero model: Llama 3.2 3B. Users: viewers who see one short screen recording, often on a phone, and the author, who runs many experiments. Peers (Neuronpedia, Goodfire, Transluce) use calm, academic looks; steering tools paint heatmaps over text. lobot shows the change itself.

**Not a dashboard:** the first screen holds one object (the fan), the model's own words, one knob and one number. No stat tiles, meter rows, sidebars of numbers, gauges or chart grids on the first screen. Evidence lives one click away.

**Mode per surface:**
- Workbench (shelf + sentence + experiment tree): Operate and tinker. One object in the main area: the tree.
- Recording mode (shelf collapsed): Persuade. The sentence and the tree only.
- Share card: Experience. A 4:5 post built from recorded text.
- Workbench (Read detail, evidence, Library, Create): Operate. Dense is allowed here, still no dashboard.

**Reference sites:** https://www.neuronpedia.org, https://www.goodfire.ai, https://transluce.org (screenshots 2026-10-29). ECG interface research (dl.gi.de, Aures et al. 2017) supports markers on the trace itself.

**Key characteristics:**
- Dark by default.
- The shared words, then curves that fork into one branch per dose.
- One rotary knob with a detent per dose; the pointer picks the branch.
- The dose in display type is the only big number.
- One pink event per screen: the split token on the selected branch, and its curve.

## Colors

**Strategy:** Committed. Ward green owns the page. Neutrals are green-tinted inks. Four pen inks carry trait data.

**Light or dark:** Dark is the default (user decision 2026-10-30): pink and ink read strongest on the dark paper in a screen recording. Light is a full token set, not an inversion, and stays available. In both, sheet is one step lighter than ground, and pink stays the brightest color on the page.

Rules for each token:
- `cut` (pink) marks one event per screen: the first token where the answer measurably changed. On the fan, that is the split token on the selected branch and that branch's curve. On the Read detail, that is where the selected behavior first crosses 1.0 (its mean). Unselected branches mark their split token in ink. Use pink nowhere else, not even for focus rings.
- `pen-1` to `pen-4` (graphite, ward blue, slate, umber) draw trait channels only. Each trait keeps one pen everywhere, and each pen always has a text label. Pens never use pink, oxblood or a state hue. All pens reach at least 5:1 on the sheet in both themes.
- `band` fills the 0 to 1 range of every channel (neutral mean to trait mean). It is a reference range, not an alarm. Values above 1 cross onto the plain sheet with no color change.
- `primary` (oxblood) is the run action only.
- `success`, `warning` and `error` signal states with a tint background and a colored mono label. No colored edges.
- `text-muted` is for metadata. Never use it to fade a model answer, except to dim text at a dose that was not recorded.

## Typography

All three faces load from Google Fonts (verified 2026-10-29: Bricolage Grotesque wdth 75 wght 800, Atkinson Hyperlegible Next 200 to 800, JetBrains Mono 400 to 700). Self-host them in the app, because lobot runs local only.

- **Bricolage Grotesque** (800, width 75): the display role has one job per screen, the live reading (`1.07`) or the dose (`+0.50`). The number is the headline; there is no tagline. The heading size (28px) sets the wordmark, section heads and the BASELINE / DOSE column heads.
- **Atkinson Hyperlegible Next:** model output (20px/1.4 on the tape, where it is the time axis; 18px in columns), UI (16px) and labels (14px bold, sentence case, no tracking).
- **JetBrains Mono** (tabular numbers): layer, neuron, dose, token index, readings in labels, timings, effect sizes and the recipe string. Every value a user might copy is mono.

Capitals appear only in the column heads (BASELINE, DOSE +1.00) and in mono state keys (COMPLETE, PASS). Levels differ by size, not only by weight.

## Layout

- Shell: a collapsible shelf (300px) on the left and the workbench beside it, capped at 1400px and centred. Collapsed, the shelf is a 56px rail with an open control and the current model and edit count. Below 860px the shelf sits on top and starts collapsed.
- Shelf, top to bottom: wordmark and collapse; Models with "+ Add"; under the active model, its behaviors with their state key ("Use" and "Edit" appear on hover or focus) and "+ New"; other models with a one-line state (on disk, gated); theme. "RESEARCH MODE · GUARD OFF" shows at the top of the shelf when it is on.
- Workbench, top to bottom: the sentence (the page's h1) with Run and Share card under it; the experiment tree; the control row (knob, dose of the selected edit, one state line).
- The sentence: "On [model ▾], push [edit] at [dose] and [edit] at [dose] [+ edit] and ask “prompt”". Every bracket is a control: the model opens the model list; an edit selects it for the knob; × removes it; "+ edit" adds a behavior. With no edits it reads "with no edit". A status line says "matches run N" or "changed since the last run".
- The experiment tree: the shared words on the left at 32 to 44px; one branch per run on the right. Curves run from the end of the shared words to each branch: pink and 4px for the selected run's split, ink for the selected run and the no-edit run, muted for other runs, dashed for a planned setup that was not run. Each branch label lists its edits; clicking it loads that setup into the sentence. A planned branch shows redacted bars and NOT RUN, never invented text.
- Word inspector: clicking any word opens a 400px panel beside the tree (below the tree under 1100px).
- Below 860px the tree stacks: shared words on top, branches below, curves drop down the left edge.
- Pages that replace the workbench in the main area: Edit behavior, New behavior, Add model, Share card. The shelf stays.

## Elevation & Depth

There are no shadows and no glow. Depth comes from layers of paper: the ground carries the grid, and the sheet sits on top with a 2px ink border. The pens are ink on the sheet, never light.

## Shapes

Corners are square (0px). Use 2px only where a browser control needs it. The knob is the one circle in the system: a printed dial face with a square-ended pointer. No pills, no rounded cards, no nested cards.

## Components

- **Focus:** every control shows a 3px `text` (ink) outline with a 2px offset. Pink never marks focus.
- **The experiment tree (the signature):** described in Layout. Every Run adds a branch where its answer first differs; tinkering grows the tree. Runs persist per prompt and model.
- **The sentence:** described in Layout. It is the only place where edits are set; there is no separate edit rack.
- **Word inspector:** follows the token inspector contract (design decision 14A): opens on click, Enter or Space on a word; Close and Esc; shows the edits applied in that run, "Observed projection" separate from the applied dose (n/a when not measured), and the top neurons at that token with their activation and a one-line "also fires on". Each neuron has Push (adds it to the sentence at +1.00) or "In edits". Note: "Neurons often mix several jobs."
- **Channel strip (Read detail, workbench):** one stepped pen line per trait, one step per token, over the `band`. Scale: 0 = neutral mean, 1 = trait mean (projection normalized by the contrast-set mean gap). Read draws a single pen line. The label says "observed projection", never "prediction". Missing values show as a gap, not a zero.
- **Live reading:** the selected trait's value at the newest token, in display type, with its token index and scale note in mono. A second trait shows at heading size.
- **Knob:** a 150 to 220px printed dial: ink ring, minor ticks, one major tick per dose step with its signed label, a heavier tick at 0. It turns the dose of the edit selected in the sentence; with no edit it dims. The square pointer snaps to a dose step. Dragging, arrow keys, Home and End all work. `role="slider"` with `aria-valuetext` set to the signed dose. Turning the knob changes the setup; it never starts generation. If the setup matches a run, that run is selected; otherwise a planned branch appears.
- **Verb line grammar:** the four verbs appear as stroke styles, never as icons. Create: a dotted line that turns solid one stage at a time. Read: a single pen line. Update: a double line, faint baseline pen plus bold steered pen; the gap is the effect. Delete: a hatched excision bracket that cuts the line, with the side effects printed inside it. The same strokes appear in the Library, in tabs and in evidence rows.
- **The pink event:** a 3px vertical rule at the event token, through the strip and the word, plus the word in `cut` and a mono flag ("crosses trait mean · token 14 · 1.07"). On Update the rule marks the split in the steered column, and the baseline token gets a 3px pink underline. Only the selected dose gets pink. Other doses mark their split in ink.
- **Run (primary):** oxblood fill, `on-primary` bold text, `⌘ Enter` hint. Disabled while a run is active; Stop takes its place.
- **Ghost button:** transparent, 2px ink border, 44px minimum height. Hover and `aria-pressed="true"` invert to ink fill.
- **Field:** no box. A 2px ink underline. Invalid: `error` underline and text, `aria-invalid`.
- **Verdict line:** a mono state key (KNOB BETTER, PROMPT BETTER, NO CLEAR WINNER, EVIDENCE UNAVAILABLE), the scope "Held-out evaluation · trait vN · dose d", and the effect delta with its interval. Withheld while streaming; STALE after a change.
- **Evidence table:** rows Read / Increase / Decrease / Delete; columns state key, effect with 95% CI, control bound, side effects, tested doses. Folded on the first screen; open in Workbench. A failed verb disables its control, with the reason inline.
- **Run states:** RUNNING (token count, elapsed time), COMPLETE, NO SPLIT, STOPPED at token N, FAILED (cause), NOT RUN.
- **Library:** a ruled list. Saved traits first (name at heading size, model, the four verb strokes with state, version), then "Create new version" per supported family.
- **Empty state:** a dashed ink box with one instruction sentence.
- **Refusal guard (design decision 6A):** a state key next to Run, re-checked on every change of the edit set: GUARD ON · NOT CHECKED, CHECKING EDITS, CHECK PASSED, BLOCKED (a note names the edit set and the measured refusal drop against the limit, with Edit doses and Research mode…), CHECK FAILED (Run stays off; Check again). Research mode asks to confirm in a dialog, lasts for the session, shows "RESEARCH MODE · GUARD OFF" above the sentence, and is recorded in every run and on the share card. It never turns on by itself.
- **Create progress (design decision 3A):** the estimate before start ("about 10-20 min, estimated"); six stage rows as an ordered list with `aria-current="step"` (Contrast prompts, Capture, Tune, Freeze, Final check, Report card), each with its work units and elapsed mono time; a persistent Stop. Stop before Freeze discards at no cost; Stop during the final check asks to confirm that this version's final check is used up. While Create runs, Run is off with the reason "Create running, stage N of 6". The server keeps working if the page closes, and the page reattaches.
- **Empty states (design decision 5A):** a model with no validated behavior shows "No validated behavior for this model yet" and one action, "Create one from a family". An experiment with no runs explains what Run will add.
- **Shelf rows:** a behavior row shows name, version and one state key (PASS, DECREASE FAIL, READ LATER, GUARD). A guard-blocked behavior is dimmed and cannot be used. Models that are gated or not downloaded are dimmed with their reason.
- **Edit behavior:** name; versions (use an older one, or create a new one that keeps the current one until it passes); the evidence table; Remove behavior with a confirmation that names what is removed. Editing never changes a direction by hand: that would void its evidence.
- **Add model:** Hugging Face id, then checks for size and memory, access, architecture support and behaviors; "Download and add"; installed models with Reload and Remove.
- **Share card:** 4:5 for an X post, built only from recorded text: wordmark, model and its edits, the prompt in quotes, the shared words forking into "no edit" and the edited answer, the dose in display type, and "same prompt · same seed · split at token N". A ready post text sits next to it. Excerpts end with an ellipsis.
- **Video storyboard (the key 10 seconds):** frame 1, one answer at ±0.00; frame 2, a hand turns the knob and the answer re-runs; frame 3, the answer forks, with pink on the steered branch. Voice-over: "Same prompt, same seed. I turn one knob inside the model, and its answer forks here."

## Do's and Don'ts

- Do: keep exactly one pink event per screen.
- Do: keep the main area to one object: the experiment tree, with the words, one knob and one number.
- Do: label every run with its true edits. A neuron recipe is never shown as a behavior.
- Do: draw every trait value as a pen step under its own word.
- Do: show signed doses with two decimals in mono (`+1.00`, `−1.00`).
- Do: label illustrative or unrecorded values on screen.
- Do: design loading, empty, stopped, error and long-output states for the tape and the knob.
- Don't: build a dashboard: no stat tiles, meter rows, gauges or number sidebars on the first screen.
- Don't: paint heatmaps over the words. The words stay ink on paper.
- Don't: use clinical or personality words in the UI: say MODEL (not PATIENT), behavior (not personality or trait profile), Dark and Light (not ward), and avoid vitals, patient, diagnosis and theatre. The look may stay clinical; the words must not suggest medical or personality analysis.
- Don't: draw the dial as a car gauge: no colored arcs, no fills, no glow.
- Don't: let the knob start generation; it shows doses that were run.
- Don't: add shadows, glow, gradients, rounded cards, brains or glitch effects.
- Don't: use pink for buttons, links, focus or decoration.

## Motion

- **Approach:** intentional, paced by data.
- **Easing:** enter ease-out, exit ease-in, move ease-in-out.
- **Duration:** micro 50 to 100ms, short 150 to 250ms, medium 250 to 400ms, long 400 to 700ms.
- **Pens:** step one token at a time as the real stream arrives. The pace comes from the model, never from a timer, except in a replay that says it is a replay.
- **Knob:** the pointer moves to its detent in 160ms ease-in-out; the sentence and the tree follow.
- **Shelf:** collapses and opens in 250ms ease-in-out.
- **The one authored moment:** the fork. When a run reaches its split, its branch curve draws from the shared words in 420ms and the split token turns pink. With `prefers-reduced-motion`, show the final state without animation.

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-29 | Initial design system created | Created by /design-consultation from the README, the office-hours design doc and screenshots of three interp sites. Three independent drafts converged on a clinical document with one hot mark at the split. |
| 2026-10-29 | Tokens taken from the approved HTML preview | AI mockups were not generated: the gstack designer has no OpenAI API key. |
| 2026-10-29 | Focus rings in ink, not pink | /design-html lint: pink measured 2.3:1 on the ground; focus needs 3:1. |
| 2026-10-29 | Labels in sentence case, untracked, 14px | /design-html lint blocks tracked uppercase labels. |
| 2026-10-29 | `on-primary` #FBF1EF, dark hover #9A3240 | Lint flags pure white as untinted paper; the old dark hover was 4.4:1 against white text. |
| 2026-10-29 | Workbench layout exceptions | The run fills the first screen, model output may exceed the word budget per screen, and the bench uses no marketing section gaps. |
| 2026-10-29 | Update for the Read era: "the ward instrument" | /design-consultation (Update mode). The product moved to behavior CRUD with Read as the headline, and the wave-1 prototype was judged too dense, with no knob and no "whoa". New memory line "I could see it, and I could turn it." Strip-chart channels under every word, one rotary knob, the reading as the only display type, verb line grammar, and pink as the first measurable change. An independent Claude subagent proposed the same core; Codex was not installed. |
| 2026-10-29 | Pen inks graphite, ward blue, slate, umber | Fulfils design decision 11A (trait palette). Umber replaces a proposed iodine #A8610A that sat too close to the warning amber. |
| 2026-10-29 | The knob snaps to recorded doses only | Keeps "moving a dose changes the recipe; generation runs on Run". The subagent proposed live re-runs; rejected for honesty and cost. |
| 2026-10-29 | The agreement trace is replaced by the channel strip and the pink rule | The tape shows agreement and divergence per token in place; a separate 72px trace repeated it. |
| 2026-10-30 | The fork: dose fan as the first screen | /design-consultation rounds 2-3. The user chose M (dose fan) as the main screen, O (storyboard) for the video and P (share card) for X. Rule: the product must not look like a dashboard. |
| 2026-10-30 | Dark by default | User preference after seeing both themes; pink and ink read strongest on dark paper in a recording. |
| 2026-10-30 | Plain words in the UI | Avoids a misread as medical advice or a personality analyzer: MODEL, behavior, Dark and Light. The clinical look stays. |
| 2026-10-30 | Channel strip moves to the workbench | Per-token pens stay as the Read detail; the first screen shows the fork instead. |
| 2026-10-30 | Workbench: H1 shelf + sentence, collapsible | User chose H1 with a collapsible shelf: the shelf manages models and behaviors (add, edit, remove); the sentence sets model and edits; collapsed, the screen becomes the recording view. |
| 2026-10-30 | Tinkering: experiment tree and word inspector | User chose T3 and T1 over an edit rack and neuron search. Every run grows a branch; clicking a word shows top neurons with Push. Follows design decision 14A. |
| 2026-10-30 | Runs carry their true recipe | The recorded run is a two-neuron recipe; it is labelled by its neurons, not as a behavior. |
