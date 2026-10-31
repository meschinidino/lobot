# Responsible use

lobot changes how an open model behaves. The same tool that adds a
cheerful voice can weaken a refusal. These rules keep it on the safe
side.

## What lobot is for

- Studying how a behavior lives inside an open model.
- Adding or removing style and persona traits, with proof.
- Checking a patch before you install or share it.

## What lobot refuses

1. **Safety-weakening descriptions.** The draft flow and the agent
   refuse descriptions that ask to remove refusals or safety behavior.
2. **Unsafe patches in the registry.** Publish refuses a patch when the
   guard blocks it or would block it, when it was made in research mode,
   or when it has no guard record or no shipped evidence.
3. **Changed patches.** Install refuses a patch whose bytes, evidence,
   signature, or publisher key do not match.

## The guard

- **Screens:** refusal (default), honesty (default), privacy (default),
  harmful compliance (opt-in: its sensitivity is not proven yet).
- **How it decides:** the same prompts run with and without the patch.
  A screen flags the patch when the patched model does worse, for example
  when it leaks a planted secret that the unedited model kept.
- **Research mode:** you can turn enforcement off to study a behavior.
  Every artifact records that, and such patches cannot be published.

## Limits (measured, not hidden)

- The screens use fixed prompt sets. A patch can pass them and still
  misbehave on other prompts.
- The local judge is a 3B model. It cannot judge some traits (for
  example rhyme). Reports say when a judge was not validated.
- Patches are tested on one model and revision. The fingerprint says
  which; re-verify on your own machine before you trust one.

## Report a problem

If a patch passes the guard but weakens safety, open an issue with the
patch and the prompt. Do not post harmful output in the issue.
