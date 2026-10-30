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
