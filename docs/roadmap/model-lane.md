# Model lane bookings

Only one agent may hold real weights at a time on the Mac. Add a `start` line
before loading a model and an `end` line after unloading it.

Format: `YYYY-MM-DD HH:MM ET  <agent>  <spec ID>  <model id>  start|end  <note>`

<!-- newest at the bottom -->
2026-10-29 22:42 ET  backend  R3  meta-llama/Llama-3.2-3B-Instruct  start  fires_on cache + demo-neuron search
2026-10-29 23:43 ET  backend  R3  meta-llama/Llama-3.2-3B-Instruct  end  shortlist + cards done, 57 passing
2026-10-29 23:45 ET  backend  Create  meta-llama/Llama-3.2-3B-Instruct  start  pirate trait for evidence page
2026-10-30 00:35 ET  backend  Create  meta-llama/Llama-3.2-3B-Instruct  end  done in 49 min, verdict knob L17@1.5
2026-10-30 00:35 ET  backend  S1  meta-llama/Llama-3.2-3B-Instruct  start  held-out guard measurement
2026-10-30 00:36 ET  backend  S1  meta-llama/Llama-3.2-3B-Instruct  end  0 misses, 1 false block, vacuous TP set
2026-10-30 00:36 ET  backend  P1  meta-llama/Llama-3.2-3B-Instruct  start  patch export + verify on pirate knob
2026-10-30 00:37 ET  backend  P1  meta-llama/Llama-3.2-3B-Instruct  end  export ok, verify PASS, no llama.cpp here
2026-10-30 00:37 ET  backend  rec  meta-llama/Llama-3.2-3B-Instruct  start  sweeps for storm neuron + pirate dial + D1 check
2026-10-30 00:44 ET  backend  rec  meta-llama/Llama-3.2-3B-Instruct  end  2 sweeps + dial means 0.53 vs 16.56
2026-10-30 00:45 ET  backend  smoke  meta-llama/Llama-3.2-3B-Instruct  start  live + replay UI smoke test
2026-10-30 00:57 ET  backend  smoke  meta-llama/Llama-3.2-3B-Instruct  end  proof line live, 6 live + 1 replay findings
2026-10-30 09:22 ET  backend  S1fu  meta-llama/Llama-3.2-3B-Instruct  start  storm-neuron flip analysis
2026-10-30 09:27 ET  backend  S1fu  meta-llama/Llama-3.2-3B-Instruct  end  real refusal loss, no rule change
2026-10-30 09:27 ET  backend  C2  meta-llama/Llama-3.2-3B-Instruct  start  hostility judge validation
2026-10-30 09:28 ET  backend  C2  meta-llama/Llama-3.2-3B-Instruct  end  hostility validated 1.000 (12/12)
2026-10-30 09:29 ET  backend  C2  meta-llama/Llama-3.2-3B-Instruct  start  hostility Create attempt 1 (run_trait seed 0)
2026-10-30 10:01 ET  backend  C2  meta-llama/Llama-3.2-3B-Instruct  end  attempt 1 KeyError BASELINE_INSTRUCTIONS (fixed, +fail-fast +coverage test)
2026-10-30 10:03 ET  backend  C2  meta-llama/Llama-3.2-3B-Instruct  start  hostility Create attempt 2 (run_trait seed 0)
2026-10-30 10:21 ET  backend  C2  -  end  attempt 2 stopped per user; Prove beat keeps pirate, no second Create
2026-10-30 10:32 ET  backend  REC  meta-llama/Llama-3.2-3B-Instruct  start  L11N1343 verified-guard check 0.0/2.0
2026-10-30 10:38 ET  backend  REC  meta-llama/Llama-3.2-3B-Instruct  end  L11 0.0 passes, 2.0 blocks verified
2026-10-30 10:38 ET  backend  REC  meta-llama/Llama-3.2-3B-Instruct  start  storm sweep 0/1.0/2.0
2026-10-30 10:44 ET  backend  REC  meta-llama/Llama-3.2-3B-Instruct  end  storm sweep recorded (22.7s model time)
2026-10-30 10:44 ET  backend  REC  meta-llama/Llama-3.2-3B-Instruct  start  pirate-dial sweep 0.2-1.0
2026-10-30 10:50 ET  backend  REC  meta-llama/Llama-3.2-3B-Instruct  end  both sweeps recorded (13.8s + 16.7s)
2026-10-30 11:05 ET  backend  smoke  meta-llama/Llama-3.2-3B-Instruct  start  day-2 live UI pass
2026-10-30 11:40 ET  backend  smoke  meta-llama/Llama-3.2-3B-Instruct  end  live pass done (proof 17550x, B2-B5 fixed, new D1-D3)
2026-10-30 11:42 ET  backend  smoke  replay  start  day-2 replay UI pass
2026-10-30 11:55 ET  backend  smoke  replay  end  replay pass done (R1 fixed, card opens)
2026-10-30 11:10 ET  backend  verify  meta-llama/Llama-3.2-3B-Instruct  start  D1/D2/D3 + L11 UI beats
2026-10-30 11:18 ET  backend  verify  meta-llama/Llama-3.2-3B-Instruct  end  D1/D2/D3 fixed, L11 beats run in UI
2026-10-30 12:27 ET  backend  demo  Qwen/Qwen3.5-4B  start  live server for Dino to click through
2026-10-31 00:22 ET  backend  F1  meta-llama/Llama-3.2-3B-Instruct  start  SAE act cache + train L12,L20 (lane free: no server on :7860; demo line above has no end)
2026-10-31 00:51 ET  backend  F1  meta-llama/Llama-3.2-3B-Instruct  end  L12+L20 trained, eval OVERALL PASS, 3 cards printed
2026-10-31 00:52 ET  backend  F1  meta-llama/Llama-3.2-3B-Instruct  start  recalibrate L12,L17,L20 under current preset key (cache merge)
2026-10-31 00:53 ET  backend  F1  meta-llama/Llama-3.2-3B-Instruct  end  cache holds L12,L17,L20 under key 403630c563648d1a
2026-10-31 01:02 ET  backend  C2  meta-llama/Llama-3.2-3B-Instruct  start  blunt validation + fast/full matrix seed 0 (7 families)
2026-10-31 03:55 ET  backend  C2  meta-llama/Llama-3.2-3B-Instruct  end  blunt 0.750 unvalidated; 6 fast + 1 full + 1 repeat done; both gates FAIL (see reports/C2.md)
2026-10-31 04:03 ET  backend  C3  meta-llama/Llama-3.2-3B-Instruct  start  live server: draft rhyming + noir traits, label, validate
2026-10-31 06:02 ET  backend  C3  meta-llama/Llama-3.2-3B-Instruct  end  rhyming 0.55 FAIL, noir 0.85 + knob; server stopped; C3 gate FAILS 1/2
2026-10-31 06:07 ET  backend  CH  meta-llama/Llama-3.2-3B-Instruct  start  10-turn replay + dial-equality check
2026-10-31 06:20 ET  backend  CH  meta-llama/Llama-3.2-3B-Instruct  end  check_chat_replay OVERALL PASS (10/10, dial diff 0.0)
2026-10-31 06:26 ET  backend  DL  meta-llama/Llama-3.2-3B-Instruct  start  live /api/delete + /api/delete/check on pirate
2026-10-31 06:33 ET  backend  DL  meta-llama/Llama-3.2-3B-Instruct  end  pirate card PASS (mean 0.360), NLL +0.035, refusal 7/8 both, guard verified; 205.7s
2026-10-31 06:39 ET  backend  IN  meta-llama/Llama-3.2-3B-Instruct  start  live install pirate into llama.cpp + judged verify
2026-10-31 06:43 ET  backend  IN  meta-llama/Llama-3.2-3B-Instruct  end  pirate installed into llama.cpp; differs + 8/8 recorded tokens; interval miss x2 (see reports/IN.md)
2026-10-31 06:52 ET  backend  RV  meta-llama/Llama-3.2-3B-Instruct  start  live POST /api/verify pirate quick + script parity
2026-10-31 07:02 ET  backend  RV  meta-llama/Llama-3.2-3B-Instruct  end  pirate quick PASS 0.745 in [0.728,0.812] via route, upload, script (identical)
2026-10-31 07:21 ET  backend  GX  meta-llama/Llama-3.2-3B-Instruct  start  gate_guard tuning (3 new screens) + held-out
2026-10-31 08:14 ET  backend  GX  meta-llama/Llama-3.2-3B-Instruct  end  tuning + held-out all screens: 0 misses everywhere; honesty/privacy caught flips; harmful vacuous (opt-in)
2026-10-31 07:42 ET  backend  CB  meta-llama/Llama-3.2-3B-Instruct  start  live pair matrix (cheerful/shakespeare/noir) + proof pair hunt
2026-10-31 08:05 ET  backend  CB  meta-llama/Llama-3.2-3B-Instruct  end  proof shakespeare+noir: singles pass honesty, union flips (0.442); cache bug fixed; synthetic halves retired
2026-10-31 11:49 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  lobot-llamad smoke (info+prefill+generate short)
2026-10-31 11:52 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  daemon info+prefill+generate smoke PASS (28/3072/128256, fast Metal timings)
2026-10-31 11:58 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  GGUF convert F16+Q8_0 with pinned toolchain
2026-10-31 12:09 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  F16+Q8_0 converted with pinned toolchain + sidecars
2026-10-31 12:09 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  F16 parity + speed + guard script
2026-10-31 12:46 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  F16 PASS (12/12 exact) + Q8_0 PASS (100% top1, verdicts identical)
2026-10-31 12:46 ET  muse  QWEN  Qwen/Qwen3.5-4B  start  GGUF convert F16+Q8_0 + parity
2026-10-31 13:03 ET  muse  QWEN  Qwen/Qwen3.5-4B  end  F16 PASS (12/12 exact) + Q8_0 PASS (10/12 exact, bars met)
2026-10-31 15:49 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  load-time compare torch vs daemon
2026-10-31 15:50 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  torch 6.8s vs daemon F16 2.6s / Q8 1.4s
2026-10-31 15:50 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  timed fast Create on torch (no-regression probe)
2026-10-31 15:52 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  fast pirate create 45s wall (no_knob verdict, cache hit)
2026-10-31 15:54 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  before-tree timed create (HEAD worktree)
2026-10-31 15:58 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  before=45s now=45s, identical
2026-10-31 17:30 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  new fast Create cheerful/Llama
2026-10-31 17:35 ET  muse  LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  cheerful fast 45s no_knob (strong tune effects, gate rejects)
2026-10-31 17:38 ET  muse  QWEN  Qwen/Qwen3.5-4B  end  ship smoke: torch + llama boots, runs complete, dial 501
2026-10-31 18:16 ET  claude  Q4KM  meta-llama/Llama-3.2-3B-Instruct  start  Q4_K_M convert + parity (Llama, then Qwen)
2026-10-31 18:21 ET  claude  Q4KM  Qwen/Qwen3.5-4B  end  Q4_K_M fails Q8 bar on both (Llama top-1 2/3, Qwen refusal 0.034)
2026-10-31 18:21 ET  claude  PREFIX  meta-llama/Llama-3.2-3B-Instruct  start  prefix cache cached-vs-cold check (Llama, then Qwen; Q8_0)
2026-10-31 18:31 ET  claude  PREFIX  Qwen/Qwen3.5-4B  end  cached==cold on both (24/24 turns exact); Llama TTFT 686->56ms @620 tok; Qwen no reuse (recurrent)
2026-10-31 18:26 ET  claude  PREFIX  Qwen/Qwen3.5-4B  start  state checkpoints: cached-vs-cold check (Qwen, then Llama; Q8_0)
2026-10-31 18:28 ET  claude  PREFIX  Qwen/Qwen3.5-4B  end  checkpoints: Qwen 12/12 exact, TTFT 825->215ms @594 tok; Llama unchanged
2026-10-31 18:48 ET  claude  CLANE  meta-llama/Llama-3.2-3B-Instruct  start  batch ops sanity (generate_batch/score_batch vs singles)
2026-10-31 18:54 ET  claude  CLANE  meta-llama/Llama-3.2-3B-Instruct  start  Create lane vs torch equality (gen/judge/nll, pirate tuning)
2026-10-31 18:59 ET  claude  CLANE  meta-llama/Llama-3.2-3B-Instruct  start  full pirate Create: llama lane, then HEAD torch reference
2026-10-31 19:12 ET  claude  CLANE  meta-llama/Llama-3.2-3B-Instruct  end  full pirate Create before/after done (logs/clane_pirate_*)
2026-10-31 19:14 ET  claude  C3FIX  meta-llama/Llama-3.2-3B-Instruct  start  redraft rhyming with judge-phrase meta-prompt + 20 replies
2026-10-31 19:18 ET  claude  C3FIX  meta-llama/Llama-3.2-3B-Instruct  start  cowboy full Create (llama lane), then HEAD torch pirate reference
2026-10-31 19:20 ET  claude  C3FIX  meta-llama/Llama-3.2-3B-Instruct  start  cowboy full Create (llama lane), then HEAD torch pirate reference
2026-10-31 19:26 ET  claude  C3FIX  meta-llama/Llama-3.2-3B-Instruct  start  cowboy full Create (llama lane), then HEAD torch pirate reference
2026-10-31 19:48 ET  claude  GUARDFP  meta-llama/Llama-3.2-3B-Instruct  start  cowboy privacy-screen texts
2026-10-31 19:49 ET  claude  GUARDFP  meta-llama/Llama-3.2-3B-Instruct  end  not a false positive: cowboy L17@0.5 leaks 4 planted secrets the base keeps (door code, user id, password, combination)
2026-10-31 19:53 ET  claude  GUARD  meta-llama/Llama-3.2-3B-Instruct  start  confirm guard (full generation) on 5 patches
2026-10-31 19:57 ET  claude  GUARD  meta-llama/Llama-3.2-3B-Instruct  end  guard confirmations in scratchpad/guarded
2026-10-31 19:57 ET  claude  AGENT  meta-llama/Llama-3.2-3B-Instruct  start  live agent: excited sports commentator (Muse + llama lane)
2026-10-31 20:06 ET  claude  AGENT  meta-llama/Llama-3.2-3B-Instruct  end  live agent run
2026-10-31 20:06 ET  claude  GUARD  meta-llama/Llama-3.2-3B-Instruct  start  privacy screen random-direction control
2026-10-31 20:09 ET  claude  GUARD  meta-llama/Llama-3.2-3B-Instruct  end  privacy random control
2026-10-31 20:25 ET  claude  GUARD2  meta-llama/Llama-3.2-3B-Instruct  start  re-confirm 5 patches with privacy v2
2026-10-31 20:43 ET  claude  GUARD2  meta-llama/Llama-3.2-3B-Instruct  end  privacy v2 confirmations
2026-10-31 20:43 ET  claude  CI  meta-llama/Llama-3.2-3B-Instruct  start  behavior CI: torch build
2026-10-31 20:44 ET  claude  CI  meta-llama/Llama-3.2-3B-Instruct  end  torch build
2026-10-31 20:44 ET  claude  CI  meta-llama/Llama-3.2-3B-Instruct  start  behavior CI: llama q4_k_m build
2026-10-31 20:45 ET  claude  CI  meta-llama/Llama-3.2-3B-Instruct  end  q4_k_m build
2026-10-31 20:45 ET  claude  XM  Qwen/Qwen3.5-4B  start  pirate transfer Llama -> Qwen (llama lane q8_0)
2026-10-31 20:57 ET  claude  XM  Qwen/Qwen3.5-4B  end  pirate transfer
2026-10-31 20:57 ET  claude  HEADREF  meta-llama/Llama-3.2-3B-Instruct  start  torch HEAD pirate full Create reference
2026-10-31 21:31 ET  claude  HEADREF  meta-llama/Llama-3.2-3B-Instruct  end  torch HEAD reference
2026-11-01 03:10 ET  claude  PP+GR  meta-llama/Llama-3.2-3B-Instruct  start  Pareto pirate+cowboy, repair pirate+noir (llama lane f16)
2026-11-01 04:09 ET  claude  PP+GR  meta-llama/Llama-3.2-3B-Instruct  end  Pareto v1+v2, repair project+gradient
2026-11-01 09:53 ET  claude  FD  meta-llama/Llama-3.2-3B-Instruct  start  fluency-aware dose: pirate+cowboy Create --pareto (llama lane f16)
2026-11-01 10:17 ET  claude  FD  meta-llama/Llama-3.2-3B-Instruct  end  fluency-aware dose pirate+cowboy
2026-11-01 19:07 ET  claude  UX2  meta-llama/Llama-3.2-3B-Instruct  start  live UI check: chat, delete, re-check, Muse agent (server, llama lane)
2026-11-01 20:17 ET  claude  UX2  meta-llama/Llama-3.2-3B-Instruct  end  live UI check done; install check HOLDS after stdin fix
2026-11-01 20:20 ET  claude  UX3  meta-llama/Llama-3.2-3B-Instruct  start  live UI: removal view, labeling with Muse suggestions
2026-11-01 20:33 ET  claude  UX3  meta-llama/Llama-3.2-3B-Instruct  end  removal view (noir) and labeling with Muse suggestions done
2026-11-02 00:32 ET  muse  MOD  meta-llama/Llama-3.2-3B-Instruct  start  live parity: torch Create pirate seed 0 full schedule vs HEADREF
2026-11-02 01:28 ET  muse  MOD  meta-llama/Llama-3.2-3B-Instruct  end  live parity pirate seed 0: knob L16@0.5, matches FD ref; HEADREF stale (pre-FD rule)
2026-11-02 08:38 ET  claude  V1  meta-llama/Llama-3.2-3B-Instruct  start  demo video recording (live server, Playwright)
2026-11-02 09:09 ET  claude  V1  meta-llama/Llama-3.2-3B-Instruct  end  demo video recorded (hook, storm guard block, cowboy fork, evidence, share)
2026-11-02 10:31 ET  muse  MODREF  meta-llama/Llama-3.2-3B-Instruct  start  new torch reference: full Create pirate seed 0 with prompt comparison, via the app
2026-11-02 12:25 ET  muse  MODREF  meta-llama/Llama-3.2-3B-Instruct  end  new torch reference logs/modref_pirate: knob L16@0.5 gain 0.241, pareto tradeoff, 3214.6s; app Create pirate-0-20261102-123532-000 also done (guard on, blocked, same frozen and guard numbers)
2026-11-02 15:55 ET  muse  TUNE  meta-llama/Llama-3.2-3B-Instruct  start  Tune with Muse live: pirate v5 and cowboy v2 dose tuning
2026-11-02 16:19 ET  muse  TUNE  meta-llama/Llama-3.2-3B-Instruct  end  Tune with Muse live done: pirate v5 fail (final gain 0.041, guard clean), cowboy v2 fail (final gain 0.657, full guard blocked)
2026-11-02 19:36 ET  muse  FIND  meta-llama/Llama-3.2-3B-Instruct  start  Find with Muse live: storms and pirate goals
2026-11-02 19:55 ET  muse  FIND  meta-llama/Llama-3.2-3B-Instruct  end  Find with Muse live done: storms fail (lift 354836, full guard blocked), pirate fail (lift 162205, full guard blocked)
2026-11-02 20:15 ET  muse  FIND  meta-llama/Llama-3.2-3B-Instruct  start  Find with Muse fixes live: storms goal with fluency gate
2026-11-02 20:21 ET  muse  FIND  meta-llama/Llama-3.2-3B-Instruct  end  Find fixes live done: storms fail (lift x2.18e+04, final fluency 0.896, full guard blocked)
2026-11-02 20:35 ET  muse  FIND  meta-llama/Llama-3.2-3B-Instruct  start  Find fixes round 2 live: storms, last confirmed recipe and log10 predictions
2026-11-02 20:51 ET  muse  FIND  meta-llama/Llama-3.2-3B-Instruct  end  Find fixes round 2 live done: storms fail (Muse picked L19N4330@1.2, lift x1.14e+05, fluency 0.623, full guard blocked)
2026-11-02 19:33 ET  claude  V3  meta-llama/Llama-3.2-3B-Instruct  start  demo video: Find with Muse live, storms goal (live server, Playwright)
2026-11-02 19:38 ET  claude  V3  meta-llama/Llama-3.2-3B-Instruct  end  Find with Muse live recorded: storms, Muse picked L19N4330@1.2 (2 confirmed tries), lift x1.24e+05, fluency 0.623, full guard blocked
2026-11-03 15:07 ET  muse  GUARD-BATCH  Qwen/Qwen3.5-4B  start  restart on batched guard code, time one guarded update run
2026-11-03 15:20 ET  muse  GUARD-BATCH  Qwen/Qwen3.5-4B  end  guarded update: full check 78s cold, 62s with warm natural cache (was 165s+ unfinished); verdict blocked by full guard both times
2026-11-03 15:32 ET  muse  FRESH-MEASURE  meta-llama/Llama-3.2-3B-Instruct  start  knob doses on 10 fresh prompts: gain, fluency, collapse for cowboy, pirate, shakespeare, sports
2026-11-03 15:40 ET  muse  FRESH-MEASURE  meta-llama/Llama-3.2-3B-Instruct  end  fresh: cowboy gain 0.017 weak 9/10; pirate 0.143 weak 8/10; shakespeare 0.277 weak 4/10; sports fluency 0.002 collapsed 8/10
2026-11-03 15:41 ET  muse  PARETO  meta-llama/Llama-3.2-3B-Instruct  start  pareto_job on cowboy, pirate, shakespeare, sports workdirs
2026-11-03 16:19 ET  muse  PARETO  meta-llama/Llama-3.2-3B-Instruct  end  cowboy@1.0 gain 0.720 flu 0.726 (knob 0.5: 0.184); pirate@1.0 0.882 flu 0.554 (knob 0.5: 0.365); shakespeare@1.0 flu 0.335 fail; sports@2.0 flu 0.008, @0.5 flu 0.947 gain 0.179
2026-11-03 16:24 ET  muse  FRESH-RECHECK  meta-llama/Llama-3.2-3B-Instruct  start  fresh prompts at fluency-first doses: cowboy@1.0, pirate@1.0, sports@0.5
2026-11-03 16:31 ET  muse  FRESH-RECHECK  meta-llama/Llama-3.2-3B-Instruct  end  cowboy@1.0 gain 0.703 flu 0.863; pirate@1.0 gain 0.964 flu 0.665; sports@0.5 flu 0.990 collapsed 0/10 gain 0.166
2026-11-03 16:32 ET  muse  CREATE-FV2  meta-llama/Llama-3.2-3B-Instruct  start  re-run Create with fluency-first-v2 for cowboy, pirate, shakespeare, sports, then import
2026-11-03 18:07 ET  muse  CREATE-FV2  meta-llama/Llama-3.2-3B-Instruct  note  full schedule too slow today (cowboy tuning >40 min); restart chain on fast schedule per user choice
2026-11-03 18:24 ET  muse  CREATE-FV2  meta-llama/Llama-3.2-3B-Instruct  end  fast: cowboy knob L20@1.0, shakespeare knob L20@1.0; pirate and sports no_knob (fast grid has no 0.5 dose; at 1.0 fluency collapses on tuning prompts)
2026-11-03 18:26 ET  muse  CREATE-FV2-FULL  meta-llama/Llama-3.2-3B-Instruct  start  full Create for pirate and sports with fluency-first-v2, then import
2026-11-03 22:52 ET  muse  CREATE-FV2-FULL  meta-llama/Llama-3.2-3B-Instruct  end  pirate knob L16@1.0, sports knob L17@0.5, both imported
2026-11-03 23:05 ET  muse  WEEKEND-REPRO  meta-llama/Llama-3.2-3B-Instruct  start  reproduce weekend + screenshot prompts at fv2 knobs, score gain/fluency/collapse
2026-11-03 23:14 ET  muse  WEEKEND-REPRO  meta-llama/Llama-3.2-3B-Instruct  end  weekend+learn prompts: cowboy gain .72/.96 flu .99/.99; pirate .69/.96 flu .22/.63 (arrr tail); shakespeare .79/.85 flu .005/.11 (word loops); sports .46/.42 flu 1.0/1.0
2026-11-03 23:22 ET  muse  CREATE-FV3  meta-llama/Llama-3.2-3B-Instruct  start  full Create with fluency-first-v3 for pirate and shakespeare, then import and re-measure
2026-11-04 02:47 ET  muse  CREATE-FV3  meta-llama/Llama-3.2-3B-Instruct  end  pirate knob L16@0.5, shakespeare knob L13@0.5, both imported; fresh10: cowboy .653/.848, pirate .151/.982 weak 7/10, shakespeare .623/.981, sports .346/.996
2026-11-04 12:16 ET  muse  EVIL-CHECK  meta-llama/Llama-3.2-3B-Instruct  start  dose sweep for is-evil-and-mean: vector, prompt, prompt+vector on 4 prompts
2026-11-04 12:24 ET  muse  EVIL-CHECK  meta-llama/Llama-3.2-3B-Instruct  end  vector@1.0 evil judge 0.55-0.99 flu 0.003-0.63; @1.5+ collapse; prompt-only judge 0.77-0.93 flu 0.61-0.92 on all 4 prompts
2026-11-04 13:52 ET  muse  DEBUG-AGENT-QWEN  Qwen/Qwen3.5-4B  start  reproduce AcceleratorError from agent Create on Qwen
2026-11-04 14:47 ET  muse  DEBUG-AGENT-QWEN  Qwen/Qwen3.5-4B  end  plain Create and full agent CLI pass on Qwen; server crash traced to orphan guard worker thread after stream close, fixed with cancel+join
2026-11-05 12:10 ET  muse  AGENT-LLAMA  meta-llama/Llama-3.2-3B-Instruct  start  clean restart after MPS OOM (stale Qwen daemon + torch resident); agent build on llama lane
2026-11-05 12:31 ET  muse  AGENT-LLAMA  meta-llama/Llama-3.2-3B-Instruct  end  agent build on llama lane completed without OOM: family be-rude-hateful-and-mocking-2, verdict no_knob, agent stopped
2026-11-05 13:32 ET  muse  EVIL-SWEEP  meta-llama/Llama-3.2-3B-Instruct  start  top-6 layers x doses 0.5-2.0 on final split, gain+fluency
2026-11-05 13:47 ET  muse  EVIL-SWEEP  meta-llama/Llama-3.2-3B-Instruct  end  24 cells (6 layers x 4 doses): every gain>=0.30 cell has a broken prompt (worst fluency <=0.035); every clean cell gains <=0.057
2026-11-05 15:02 ET  muse  EVIL-SWEEP-ALL  meta-llama/Llama-3.2-3B-Instruct  start  all 28 layers x doses 0.5-2.0 on final split, gain+fluency; server on 7860 stopped to free the GPU slot
2026-11-05 16:40 ET  muse  EVIL-SWEEP-ALL  meta-llama/Llama-3.2-3B-Instruct  end  112 cells done; one clean strong cell L12@0.5 (gain 0.338, worst fluency 0.679) on final[:10]; needs fresh-prompt check
2026-11-05 16:40 ET  muse  EVIL-VERIFY-L12  meta-llama/Llama-3.2-3B-Instruct  start  L12 dose 0.5 on final[10:30], 20 prompts never used in the sweep
2026-11-05 16:52 ET  muse  EVIL-VERIFY-L12  meta-llama/Llama-3.2-3B-Instruct  end  L12@0.5 on 20 fresh prompts: gain 0.129 CI95 [0.04, 0.239], under the 0.30 bar; fluency mean 0.986 worst 0.876. The one clean strong sweep cell was search noise.
2026-11-05 17:10 ET  muse  ML-COMBINE  meta-llama/Llama-3.2-3B-Instruct  start  ML step1+2: layer ranking + greedy combine search on be-rude-hateful-and-mocking-2 (search tuning[:10], confirm final[0:30]); server stopped to free the slot
2026-11-05 17:35 ET  muse  ML-COMBINE  meta-llama/Llama-3.2-3B-Instruct  end  best mix L16@0.5+L27@0.5: search gain 0.343 clean; confirm on final 30 gain 0.260 (bar 0.30), fluency mean 0.926 p10 0.769: FAIL, close. Layer 12 ranked 15/28 by steer_score (spec said top half: miss by one)
2026-11-05 18:05 ET  muse  ML-COMBINE-V2  meta-llama/Llama-3.2-3B-Instruct  start  probe-ranked greedy (threshold 0.01, doses to 1.5) on be-rude-hateful-and-mocking-2; confirm on 20 brand-new hand-written questions (fresh_questions_v2.json); server stopped
2026-11-05 19:05 ET  muse  ML-COMBINE-V2  meta-llama/Llama-3.2-3B-Instruct  end  probe ranks L12 3/28 (v2 gate pass); winner single L11@0.5, search gain 0.465 clean; confirm on 20 brand-new questions gain 0.572 but fluency p10 0.026 (gate 0.3): FAIL on fluency
2026-11-06 02:20 ET  muse  ML-LEARN-V3  meta-llama/Llama-3.2-3B-Instruct  start  learned vectors on be-rude-hateful-and-mocking-2: probe top-8 layers, contrast answers on tuning, confirm on fresh_questions_v3; server stopped
2026-11-06 03:05 ET  muse  ML-LEARN-V3  meta-llama/Llama-3.2-3B-Instruct  end  trained vectors (90 steps, pos NLL 3.5->0.8, drift KL 4.4->0.28); search scale 0.75 gain 0.881; v3 confirm FAIL (gain 0.819, fluency p10 0.113); v3b confirm at scale 0.625 on fresh v4 PASS: gain 0.402, fluency mean 0.837, p10 0.392
2026-11-06 03:20 ET  muse  ML-LEARN-V3B-REPRO  meta-llama/Llama-3.2-3B-Instruct  start  shipped fixed-scale path: learn_vectors.py --vectors learned_vectors.pt --scale 0.625 --fresh fresh_questions_v4.json (reproduction of the v3b confirm through committed code)
2026-11-06 03:28 ET  muse  ML-LEARN-V3B-REPRO  meta-llama/Llama-3.2-3B-Instruct  end  shipped fixed-scale path reproduces the v3b confirm exactly: gain 0.402, fluency 0.837/0.392, PASS
2026-11-06 03:50 ET  muse  LEARNED-IN-APP  meta-llama/Llama-3.2-3B-Instruct  start  verify learned mix through the app run path with guard on (server run, Llama warm)
2026-11-06 04:05 ET  muse  LEARNED-IN-APP  meta-llama/Llama-3.2-3B-Instruct  end  learned mix runs through the app update path (research mode text shifts meaner, clean); full guard BLOCKS it at dose 1.0 (min_ratio 0.0, refusal loss) - recorded in learned.json; normal mode stays blocked, research only
2026-11-06 12:20 ET  muse  FACET-RECONFIRM  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.625 confirm of learned vectors on fresh_questions_v5 under the facet-min judge
2026-11-06 12:20 ET  muse  FACET-RECONFIRM  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.625 confirm of learned vectors on fresh_questions_v5 under the facet-min judge
2026-11-06 12:26 ET  muse  FACET-RECONFIRM  meta-llama/Llama-3.2-3B-Instruct  end  FAIL: gain 0.182 (bar 0.30), fluency mean 0.500, p10 0.101 (bar 0.30) on fresh_questions_v5; the mix is condescending, not hateful, and the strict judge sees it
2026-11-06 12:40 ET  muse  FEWSHOT-RETRAIN  meta-llama/Llama-3.2-3B-Instruct  start  retrain learned vectors with 3 few-shot rude examples on the positive side; confirm on fresh_questions_v6 under the facet-min judge
2026-11-06 13:05 ET  muse  FEWSHOT-RETRAIN  meta-llama/Llama-3.2-3B-Instruct  end  facet gain now strong (confirm at scale 1.0 on v6: gain 0.681) but fluency collapsed (mean 0.109); scale search picked max gain when no scale passed fluency. Fluency label facet-split bug found and fixed; re-confirm at 0.75 on fresh v7 next
2026-11-06 13:06 ET  muse  FEWSHOT-CONFIRM-075  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.75 confirm of few-shot vectors on fresh_questions_v7 with fixed fluency label
2026-11-06 13:12 ET  muse  FEWSHOT-CONFIRM-075  meta-llama/Llama-3.2-3B-Instruct  end  FAIL on v7: gain 0.478 (pass), fluency mean 0.555, p10 0.015 (fail). Sampled outputs are truly rude/mocking (trait up to 0.911); tail is repetition. Confirm at 0.5 on fresh v8 next
2026-11-06 13:13 ET  muse  FEWSHOT-CONFIRM-050  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.5 confirm of few-shot vectors on fresh_questions_v8
2026-11-06 13:20 ET  muse  FEWSHOT-CONFIRM-050  meta-llama/Llama-3.2-3B-Instruct  end  FAIL on v8: gain 0.004, fluency mean 0.947, p10 0.848. At 0.5 the mix does nothing on fresh prompts; the window is narrow. Confirm midpoint 0.625 on fresh v9 next
2026-11-06 13:21 ET  muse  FEWSHOT-CONFIRM-0625  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.625 confirm of few-shot vectors on fresh_questions_v9
2026-11-06 13:28 ET  muse  FEWSHOT-CONFIRM-0625  meta-llama/Llama-3.2-3B-Instruct  end  FAIL on v9, close: gain 0.270 (bar 0.30), fluency mean 0.877, p10 0.630 (clean). Confirm 0.6875 on fresh v10 next
2026-11-06 13:29 ET  muse  FEWSHOT-CONFIRM-06875  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.6875 confirm of few-shot vectors on fresh_questions_v10
2026-11-06 13:36 ET  muse  FEWSHOT-CONFIRM-06875  meta-llama/Llama-3.2-3B-Instruct  end  FAIL on v10: gain 0.400 (pass), fluency mean 0.750, p10 0.156 (fail). Window is razor thin. Last midpoint: 0.65 on fresh v11
2026-11-06 13:37 ET  muse  FEWSHOT-CONFIRM-065  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.65 confirm of few-shot vectors on fresh_questions_v11
2026-11-06 13:44 ET  muse  FEWSHOT-CONFIRM-065  meta-llama/Llama-3.2-3B-Instruct  end  FAIL on v11, double near miss: gain 0.257, fluency p10 0.288 (bars 0.30/0.30). Few-shot vectors saved as learned_vectors_fewshot1.pt. Retrain with drift_w 2.5 next; search picks a fluent scale, confirms on fresh v12
2026-11-06 13:45 ET  muse  DRIFT25-RETRAIN  meta-llama/Llama-3.2-3B-Instruct  start  retrain few-shot vectors with drift weight 2.5; scale search + confirm on fresh_questions_v12
2026-11-06 14:02 ET  muse  DRIFT25-RETRAIN  meta-llama/Llama-3.2-3B-Instruct  end  confirm at 0.75 on v12 FAIL on fluency tail only: gain 0.311 (pass), fluency mean 0.688, p10 0.202. Last scale try: 0.65 on fresh v13
2026-11-06 14:03 ET  muse  DRIFT25-CONFIRM-065  meta-llama/Llama-3.2-3B-Instruct  start  fixed-scale 0.65 confirm of drift-2.5 vectors on fresh_questions_v13
2026-11-06 14:10 ET  muse  DRIFT25-CONFIRM-065  meta-llama/Llama-3.2-3B-Instruct  end  FAIL on v13: gain 0.173, fluency p10 0.148. Scale hunt stops. Best mix stays few-shot drift-1.0 vectors (learned_vectors.pt restored); no fresh confirm passes both bars yet
2026-11-08 12:37 ET  muse  SH-B5-RETRAIN90  meta-llama/Llama-3.2-3B-Instruct  start  retrain rude learned vectors on 90 questions (30 tuning + 60 extra) with --shots; scale search, confirm on fresh_questions_v14 (50 brand-new)
2026-11-08 13:30 ET  muse  SH-B5-RETRAIN90  meta-llama/Llama-3.2-3B-Instruct  end  FAIL on fresh_questions_v14 (50 brand-new): best scale 1.0, gain 0.848 (bar 0.30), fluency mean 0.588, fluency p10 0.202 (bar 0.30), min 0.004. 90 training questions fixed strength, not the tail. Step 6 dose cap next; v14 is spent
