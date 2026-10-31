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
