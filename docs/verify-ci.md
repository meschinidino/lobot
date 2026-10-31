# Re-verify a shipped patch (third-party run)

A `.lobot` patch ships its evidence: the trait, the frozen
knob, and the report card with the shipped 95% interval.
Anyone can re-run that evidence on their own machine. PASS
means the measured effect lands inside the shipped interval.

## One command

1. Check out the repo and install `uv`.
2. Fetch the patch file (a `knob.lobot` from a trait job).
3. Run the check:

```bash
uv run python scripts/verify_patch.py knob.lobot --quick
```

`--quick` checks 16 held-out prompts. Without it, the full
30-prompt final split runs. Exit 0 is PASS, 1 is FAIL, 2 is
a rejected or unreadable patch.

## Weights

The check loads the model named in the patch fingerprint.

- Qwen (`Qwen/Qwen3.5-4B`) is public: no token needed.
- Llama (`meta-llama/Llama-3.2-3B-Instruct`) is gated:
  accept the license on Hugging Face, then export
  `HF_TOKEN` before the run. Never print the token.

## In CI

The workflow file ships staged at `docs/ci/verify-patch.yml`
(the automation token lacks the `workflow` scope, so a
maintainer must copy it to `.github/workflows/` once).
Then call the reusable workflow from your repo:

```yaml
jobs:
  reverify:
    uses: meschinidino/lobot/.github/workflows/verify-patch.yml@main
    with:
      patch_path: patches/pirate-v1.lobot
      quick: true
    secrets:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

The job fails when the patch FAILs. GPU runners are not
required; CPU runs work and take longer.

## As a UI flow

`POST /api/verify {"trait_id", "quick"}` streams per-prompt
progress then the result for a saved trait patch.
`POST /api/verify/upload` (multipart `.lobot`, at most
64 MB) runs the same check on an uploaded file. Both need
the patch model to be the active model (else 409). Full
shapes are in `docs/api.md`.
