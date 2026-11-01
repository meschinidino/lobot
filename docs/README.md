# lobot docs

Start with the project [README](../README.md). Coding agents start with
[AGENTS.md](../AGENTS.md).

## Guides (how to do a task)

| Guide | For |
|---|---|
| [guides/live-mode.md](guides/live-mode.md) | Run the app with real weights |
| [guides/replay-mode.md](guides/replay-mode.md) | Run the app with no weights, from recorded runs |
| [guides/create.md](guides/create.md) | Turn a behavior into a tested patch (incl. the Muse Spark agent) |
| [guides/share.md](guides/share.md) | Install, verify, publish, audit and transfer patches |
| [guides/verify-ci.md](guides/verify-ci.md) | Re-check a shipped patch in CI |
| [guides/llama-backend.md](guides/llama-backend.md) | Run replies and Create on llama.cpp |

## Reference

| Doc | Holds |
|---|---|
| [reference/api.md](reference/api.md) | Every HTTP route, request and response |
| [reference/configuration.md](reference/configuration.md) | Every environment variable and its default |
| [reference/scripts.md](reference/scripts.md) | Every script, grouped by task |
| [../presets/README.md](../presets/README.md) | Recipe string grammar and preset schema |
| [../RESPONSIBLE_USE.md](../RESPONSIBLE_USE.md) | What lobot refuses, and why |

## Design and plans

| Doc | Holds |
|---|---|
| [DESIGN.md](DESIGN.md) | The UI design system ("the fork" workbench) |
| [plans/lobot-10x-plan.md](plans/lobot-10x-plan.md) | The reviewed plan; decisions D1-D14 still bind |
| [plans/behavior-crud.md](plans/behavior-crud.md) | The behavior CRUD direction (Create, Read, Update, Delete) |
| [plans/llama-backend.md](plans/llama-backend.md) | llama.cpp backend design, parity and speed results |
| [agent-boot.md](agent-boot.md) | Runbook for agents that start the app for a user |

## Roadmap and history

| Doc | Holds |
|---|---|
| [roadmap/README.md](roadmap/README.md) | Stages (10X, 100X, 1000X), status tables, next steps |
| [roadmap/specs/](roadmap/specs/) | One spec per workstream |
| [roadmap/reports/](roadmap/reports/) | One report per workstream, with exact results |
| [roadmap/contracts.md](roadmap/contracts.md) | Shared routes, signatures and file formats |
| [roadmap/model-lane.md](roadmap/model-lane.md) | Who held the real weights, when |
| [TODOS.md](TODOS.md) | Deferred work, with the reason |
| [archive/](archive/) | Docs that are no longer current, kept for history |
