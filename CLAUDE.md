## Agents
Read AGENTS.md first: repo map, commands, the GPU/model-lane rules, branch and
commit rules, and where each kind of doc goes.

## Design System
Read docs/DESIGN.md before visual or UI work: it defines the fonts, colors, spacing, and
aesthetic direction. Ask the user before departing from it. When reviewing or QA-ing
UI, flag code that doesn't match docs/DESIGN.md.

## Boot Runbook
When the user asks to run, try, or open the app, follow docs/agent-boot.md.
What each screen does: docs/guides/app.md. Update it when a screen changes.

## Checking work
- UI: follow "Checking UI work" in AGENTS.md. Say which screens ran on real
  weights, which ran in replay, and which did not run.
- Live runs: take the model lane first (docs/roadmap/model-lane.md). Stop the
  live server before the test suite.
- Never turn the guard off or switch on Research mode to get a result. A
  guard block is a result.
