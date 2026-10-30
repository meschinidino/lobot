# After the demo: 100X and 1000X

Each item is a workstream with a goal, an acceptance test and its
dependencies. Write a full spec (same template as `specs/`) before starting.

## 100X (November to January)

| ID | Workstream | Goal | Acceptance | Depends on |
|---|---|---|---|---|
| F1 | Clean features | Train a sparse autoencoder or transcoder on chosen Llama layers on the Mac; read and steer features, not single neurons | On 50 random features, reader coherence median >= 0.6 (vs single-neuron median); steering a feature has lift >= 3 for its top words on at least half | R1, R3 |
| C2 | Fast describe-to-patch | Create in about 1 minute: fewer prompts, cached captures, early stop when the gate is clear | Median Create time <= 2 min on Llama; same pass/fail as the full schedule on 8 known families | Create (wave 1) |
| C3 | Open descriptions | Create from any description: the model writes contrast prompts and a judge rubric; 20 user labels validate the judge | Two new traits pass the gates with model-written parts; judge agreement >= the validated families' | C2, J1 (TODOS.md item) |
| CH | Chat with live dials | Multi-turn chat; edits persist across turns; dials per word across turns | 10-turn chat replays exactly; dials per token match /api/run reads | D1, U3 |
| DL | Delete | Run-time ablation for eligible traits, with side-effect checks | Delete report cards with side effects on prompts that elicit the trait; guard re-checked with Delete active | Engine ablation (exists) |
| IN | Install everywhere | One-click install into llama.cpp and Ollama from the UI; verify button | A patch installed from the UI changes llama.cpp output as recorded | P1 |
| RV | Re-verify anywhere | `verify_patch.py` becomes a UI flow and a CI action | A third party reproduces the shipped effect within its 95% interval | P1 |
| GX | Guards for more risks | Add honesty, privacy-leak and harmful-compliance screens to the guard | Each screen: zero misses on its held-out set; false blocks reported | S1 |
| CB | Combined patches | Check patch pairs and sets, not only single patches | A pair that passes alone and fails together is caught; matrix report for installed patches | GX |

## 1000X (2027)

| ID | Workstream | Goal | Acceptance |
|---|---|---|---|
| RG | Signed patch registry | Publish and install patches with signatures, evidence, guard record and model fingerprint; reject unsigned or tampered patches | A patch published by user A installs for user B, verifies, and fails loudly when changed |
| XM | Cross-model transfer | Carry a behavior from one model to another (feature matching or re-derivation), with evidence on both | A trait moved to a second model passes its gates there, or shows "unavailable" with the reason |
| CI | Behavior CI | Teams run behavior tests (guards, cards, regressions) on every model or patch change | A CI job fails when a model update breaks a shipped patch's evidence |
| SC | Scale | Larger open models via a shared, verified cache of features and evidence | One 70B-class model served from a cache with the same verify path |
| AU | Audit trail | Every change to a deployed model is recorded with its proof, for auditors | An auditor rebuilds the deployed model's behavior history from the records |

## Dual use

The same tool that adds honesty can remove safety. Every stage keeps: the
guard on by default, research mode recorded in every artifact, refusal-weakening
patches unpublishable to the registry, and no baked-in weight export without
the guard record (TODOS.md).
