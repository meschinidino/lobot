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
2026-10-30 09:52 ET  backend  S1fu  meta-llama/Llama-3.2-3B-Instruct  end  real refusal loss, no rule change
