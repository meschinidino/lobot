// lobot-llamad: minimal persistent llama.cpp worker for the lobot llama backend.
//
// Stdio JSON-lines protocol (docs/plans/llama-backend.md). Stdout carries
// protocol lines only; all llama.cpp logs go to stderr. Python owns
// tokenization: requests carry token ids, responses return token ids.
//
// Ops (one JSON object per stdin line):
//   {"op":"info"}
//     -> {"ok":true,"n_layer":N,"n_embd":N,"n_vocab":N,"n_ctx":N,
//         "desc":"...","commit":"..."}
//   {"op":"generate","ids":[...],"max_new":N,"stop_ids":[...],
//    "cvec_b64":"..."|null,"il_start":L,"il_end":L}
//     -> {"type":"token","id":T}* then
//        {"type":"done","ids":[...],"stop":"maxlen|stop|cancel|error",
//         "cancelled":bool,"prefill_ms":F,"decode_ms":F,
//         "n_prefill":N,"n_reused":N,"error":null|"..."}.
//        Greedy (argmax). Stops only on stop_ids. cancel works between tokens.
//   {"op":"prefill","ids":[...],"cvec_b64":"..."|null,"il_start":L,"il_end":L}
//     -> {"ok":true,"n_reused":N,"logits_b64":"..."} (full last-position
//        logits, f32 LE)
//   {"op":"generate_batch",...} and {"op":"score_batch",...}: Create lane
//     batch ops, one sequence per item (--parallel N); see the functions.
//   {"op":"cancel"} -> no reply (flags the in-flight request). A reply
//     could land after that request's last line and desync the stream.
//   {"op":"shutdown"} -> exits 0. stdin EOF also exits 0.
//
// Prefix cache (default on, --no-prefix-cache turns it off): a request
// with the same cvec as the last one reuses the KV of their shared token
// prefix; any other cvec clears the KV first. Done/prefill replies carry
// n_reused. Reuse changes ubatch shapes, so logits can differ by float
// noise from a cold run. Recurrent/hybrid models cannot roll back part of
// a sequence; for them the daemon saves one state checkpoint per prompt,
// 8 tokens before its end, and restores it when the next prompt extends
// it. cvec_b64 is a full n_layer*n_embd
// float buffer (slot L-1 per 0-based layer L); il_start/il_end are
// 0-based touched layers.

#include "llama.h"

#include "base64.hpp"
#include "json.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifndef LOBOT_LLAMACPP_COMMIT
#define LOBOT_LLAMACPP_COMMIT "unknown"
#endif

static void log_to_stderr(enum ggml_log_level level, const char * text, void * /*user*/) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
        std::fflush(stderr);
    }
}

static std::mutex g_out_mu;

static void emit_line(const std::string & s) {
    std::lock_guard<std::mutex> lock(g_out_mu);
    std::cout << s << "\n" << std::flush;
}

static int32_t argmax(const float * logits, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

struct Daemon {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t n_vocab = 0;
    int32_t n_ctx = 0;
    int32_t n_ubatch = 512;
    int32_t n_batch = 2048;
    int32_t n_seq = 1;
    std::atomic<bool> cancel_flag{false};

    bool load(const std::string & path, int32_t ctx_size, int32_t ubatch, int32_t threads,
              int32_t parallel) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 999; // full Metal offload
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) {
            return false;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ctx_size;
        cp.n_ubatch = ubatch;
        cp.n_batch = 2048;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        // Batch ops run one sequence per prompt; they share one KV
        // buffer (unified), so n_ctx bounds the sum of their lengths.
        cp.n_seq_max = (uint32_t) parallel;
        cp.kv_unified = true;
        n_seq = parallel;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            llama_model_free(model);
            model = nullptr;
            return false;
        }
        vocab = llama_model_get_vocab(model);
        n_layer = llama_model_n_layer(model);
        n_embd = llama_model_n_embd(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        n_ctx = llama_n_ctx(ctx);
        n_ubatch = ubatch;
        return true;
    }

    ~Daemon() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }

    // Returns "" on success, else an error string.
    std::string apply_cvec(const common_json & req) {
        if (!req.contains("cvec_b64") || req.at("cvec_b64").is_null()) {
            if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) {
                return "cvec clear failed";
            }
            return "";
        }
        std::string b64 = req.at("cvec_b64").get<std::string>();
        std::string raw;
        try {
            raw = base64::decode(b64);
        } catch (const std::exception & e) {
            return std::string("bad cvec_b64: ") + e.what();
        }
        if (raw.size() % sizeof(float) != 0) {
            return "cvec_b64 is not a float buffer";
        }
        size_t n_floats = raw.size() / sizeof(float);
        if (n_floats != (size_t) n_embd * (size_t) n_layer) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "cvec has %zu floats, want n_embd*n_layer = %d",
                          n_floats, n_embd * n_layer);
            return buf;
        }
        int32_t il_start = req.at("il_start").get<int>();
        int32_t il_end = req.at("il_end").get<int>();
        if (il_start < 1 || il_end < il_start || il_end >= n_layer) {
            return "il_start/il_end out of range [1,n_layer)";
        }
        if (llama_set_adapter_cvec(ctx, (const float *) raw.data(), n_floats,
                                   n_embd, il_start, il_end) != 0) {
            return "llama_set_adapter_cvec failed";
        }
        return "";
    }

    // Decode one chunk. logits_last: request logits on the final token.
    int decode_chunk(const std::vector<llama_token> & toks, int32_t n_past,
                     bool logits_last) {
        llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); i++) {
            batch.token[i] = toks[i];
            batch.pos[i] = n_past + (int32_t) i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (logits_last && i + 1 == toks.size()) ? 1 : 0;
        }
        batch.n_tokens = (int32_t) toks.size();
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        return rc;
    }

    void do_info() {
        char desc[256] = {0};
        llama_model_desc(model, desc, sizeof(desc));
        common_json resp({
            {"ok", true},
            {"n_layer", n_layer},
            {"n_embd", n_embd},
            {"n_vocab", n_vocab},
            {"n_ctx", n_ctx},
            {"desc", std::string(desc)},
            {"commit", std::string(LOBOT_LLAMACPP_COMMIT)},
        });
        emit_line(resp.dump());
    }

    // Prefix cache: the KV cache keeps the tokens of the last request
    // (prompt plus every decoded reply token). The next request reuses
    // the longest shared prefix when its cvec key is the same. A cvec
    // changes the residual at every position, so a new key clears all.
    bool prefix_cache = true;
    std::vector<llama_token> cached;     // tokens at KV positions 0..n-1
    std::string cached_key = "\x01none"; // cvec key the cache was built with

    // Checkpoint for recurrent/hybrid models (Qwen3.5), which cannot roll
    // back part of a sequence. The daemon decodes the last ckpt_tail prompt
    // tokens separately and saves the sequence state just before them:
    // chat templates change the end of the prompt (Qwen drops "<think>\n"
    // from history), so the next turn shares the prompt minus that tail.
    bool use_ckpt = false;
    int32_t ckpt_tail = 8;
    std::vector<llama_token> ckpt_toks;  // tokens the saved state covers
    std::vector<uint8_t> ckpt_state;     // llama_state_seq_get_data, seq 0

    static std::string cvec_key(const common_json & req) {
        if (!req.contains("cvec_b64") || req.at("cvec_b64").is_null()) {
            return "";
        }
        return req.at("cvec_b64").get<std::string>() + "|" +
               std::to_string(req.at("il_start").get<int>()) + "|" +
               std::to_string(req.at("il_end").get<int>());
    }

    void reset_cache() {
        llama_memory_clear(llama_get_memory(ctx), true);
        cached.clear();
    }

    void drop_ckpt() {
        ckpt_toks.clear();
        ckpt_state.clear();
    }

    void save_ckpt() {
        llama_synchronize(ctx);
        size_t n = llama_state_seq_get_size(ctx, 0);
        ckpt_state.resize(n);
        if (llama_state_seq_get_data(ctx, ckpt_state.data(), n, 0) != n) {
            drop_ckpt();
            return;
        }
        ckpt_toks = cached;
    }

    // Restore the checkpoint when it is a strict prefix of toks (the last
    // prompt token must still be decoded for its logits). Returns the
    // number of restored tokens, 0 when unusable.
    int32_t restore_ckpt(const std::vector<llama_token> & toks) {
        if (ckpt_toks.empty() || ckpt_toks.size() >= toks.size() ||
            !std::equal(ckpt_toks.begin(), ckpt_toks.end(), toks.begin())) {
            return 0;
        }
        reset_cache();
        if (llama_state_seq_set_data(ctx, ckpt_state.data(), ckpt_state.size(), 0) == 0) {
            reset_cache();
            drop_ckpt();
            return 0;
        }
        cached = ckpt_toks;
        return (int32_t) cached.size();
    }

    // Set the cvec and decode toks into the KV cache, reusing the cached
    // prefix when allowed. Leaves last-token logits ready and synchronized.
    // Returns "" on success, else an error. n_reused: prefix tokens kept.
    std::string prepare(const common_json & req,
                        const std::vector<llama_token> & toks,
                        int32_t & n_reused) {
        n_reused = 0;
        std::string key = cvec_key(req);
        if (!prefix_cache || key != cached_key) {
            reset_cache();
            drop_ckpt(); // the state was built with another cvec
            cached_key = "\x01none";
            std::string err = apply_cvec(req);
            if (!err.empty()) {
                return err;
            }
            cached_key = key;
        } else {
            size_t p = 0;
            while (p < cached.size() && p < toks.size() && cached[p] == toks[p]) {
                p++;
            }
            if (p == toks.size()) {
                p--; // re-decode the last token to get its logits
            }
            // Recurrent layers (Qwen3.5) refuse to drop a partial range;
            // then start over. A pure extension (p == cached.size()) works.
            if (llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) p, -1)) {
                cached.resize(p);
                n_reused = (int32_t) p;
            } else {
                n_reused = restore_ckpt(toks); // resets the cache when unusable
                if (n_reused == 0) {
                    reset_cache();
                }
            }
        }
        // Checkpoint position: before the last ckpt_tail tokens, and only
        // when that saves something new over what was reused.
        size_t ckpt_at = toks.size() > (size_t) ckpt_tail ? toks.size() - (size_t) ckpt_tail : 0;
        bool want_ckpt = use_ckpt && prefix_cache && ckpt_at > (size_t) n_reused;
        size_t off = (size_t) n_reused;
        while (off < toks.size()) {
            size_t end = std::min(off + (size_t) n_ubatch, toks.size());
            if (want_ckpt && off < ckpt_at && end > ckpt_at) {
                end = ckpt_at;
            }
            std::vector<llama_token> chunk(toks.begin() + off, toks.begin() + end);
            bool last = end == toks.size();
            if (decode_chunk(chunk, (int32_t) off, last) != 0) {
                reset_cache();
                drop_ckpt();
                return "llama_decode failed";
            }
            cached.insert(cached.end(), chunk.begin(), chunk.end());
            if (want_ckpt && end == ckpt_at) {
                save_ckpt();
            }
            off = end;
        }
        // Metal decodes asynchronously: wait here so prefill_ms is real.
        llama_synchronize(ctx);
        return "";
    }

    void do_prefill(const common_json & req) {
        std::vector<int> ids = req.at("ids").get<std::vector<int>>();
        if (ids.empty()) {
            emit_line(common_json({{"ok", false},
                                   {"error", "prefill needs ids"}}).dump());
            return;
        }
        if ((int32_t) ids.size() > n_ctx) {
            emit_line(common_json({{"ok", false},
                                   {"error", "prompt longer than n_ctx"}}).dump());
            return;
        }
        std::vector<llama_token> toks(ids.begin(), ids.end());
        llama_perf_context_reset(ctx);
        int32_t n_reused = 0;
        std::string err = prepare(req, toks, n_reused);
        if (!err.empty()) {
            emit_line(common_json({{"ok", false}, {"error", err}}).dump());
            return;
        }
        const float * logits = llama_get_logits(ctx);
        std::string blob((const char *) logits,
                         (size_t) n_vocab * sizeof(float));
        emit_line(common_json({{"ok", true},
                               {"n_reused", n_reused},
                               {"logits_b64", base64::encode(blob)}}).dump());
    }

    void emit_generate_error(const std::string & err) {
        emit_line(common_json({{"type", "done"}, {"ids", std::vector<int>{}},
                               {"stop", "error"}, {"cancelled", false},
                               {"prefill_ms", 0.0}, {"decode_ms", 0.0},
                               {"n_prefill", 0}, {"n_reused", 0},
                               {"error", err}}).dump());
    }

    void do_generate(const common_json & req) {
        std::vector<int> ids = req.at("ids").get<std::vector<int>>();
        int32_t max_new = req.at("max_new").get<int>();
        std::vector<int> stops =
            req.contains("stop_ids") ? req.at("stop_ids").get<std::vector<int>>()
                                     : std::vector<int>{};
        const std::unordered_set<int32_t> stop_set(stops.begin(), stops.end());
        if (ids.empty() || max_new < 1) {
            emit_generate_error("generate needs ids and max_new >= 1");
            return;
        }
        if ((int32_t) ids.size() + max_new > n_ctx) {
            emit_generate_error("prompt + max_new exceeds n_ctx");
            return;
        }
        std::vector<llama_token> toks(ids.begin(), ids.end());
        llama_perf_context_reset(ctx);
        int64_t t0 = llama_time_us();
        int32_t n_reused = 0;
        std::string err = prepare(req, toks, n_reused);
        if (!err.empty()) {
            emit_generate_error(err);
            return;
        }
        int64_t t1 = llama_time_us();
        int32_t n_past = (int32_t) toks.size();
        std::vector<int> out;
        std::string stop = "maxlen";
        for (int32_t step = 0; step < max_new; step++) {
            if (cancel_flag.load()) {
                stop = "cancel";
                break;
            }
            const float * logits = llama_get_logits(ctx);
            int32_t tok = argmax(logits, n_vocab);
            if (stop_set.count(tok)) {
                // Emit the stop token, then stop: HF generate
                // includes the eos id in its output, and parity
                // needs the same ids on both sides. It is never
                // decoded, so it never enters the prefix cache.
                stop = "stop";
                out.push_back(tok);
                emit_line(common_json({{"type", "token"}, {"id", tok}}).dump());
                break;
            }
            out.push_back(tok);
            emit_line(common_json({{"type", "token"}, {"id", tok}}).dump());
            std::vector<llama_token> one{(llama_token) tok};
            if (decode_chunk(one, n_past, true) != 0) {
                stop = "error";
                err = "llama_decode failed";
                reset_cache();
                break;
            }
            cached.push_back((llama_token) tok);
            n_past += 1;
        }
        int64_t t2 = llama_time_us();
        bool cancelled = stop == "cancel";
        emit_line(common_json({{"type", "done"},
                               {"ids", out},
                               {"stop", stop},
                               {"cancelled", cancelled},
                               {"prefill_ms", (t1 - t0) / 1000.0},
                               {"decode_ms", (t2 - t1) / 1000.0},
                               {"n_prefill", (int32_t) toks.size() - n_reused},
                               {"n_reused", n_reused},
                               {"error", stop == "error" ? err : ""}}).dump());
    }

    // -- batch ops (Create lane) ------------------------------------------------
    //
    // Both ops start and end with an empty KV cache (the prefix cache is
    // dropped), use one sequence per item, and answer with one line.

    // A batch with explicit per-token seq ids and output flags.
    struct Packed {
        std::vector<llama_token> tok;
        std::vector<llama_pos> pos;
        std::vector<llama_seq_id> seq;
        std::vector<int8_t> out;
        void add(llama_token t, llama_pos p, llama_seq_id s, bool o) {
            tok.push_back(t);
            pos.push_back(p);
            seq.push_back(s);
            out.push_back(o ? 1 : 0);
        }
        size_t size() const { return tok.size(); }
        void clear() { tok.clear(); pos.clear(); seq.clear(); out.clear(); }
    };

    int decode_packed(const Packed & pk) {
        llama_batch batch = llama_batch_init((int32_t) pk.size(), 0, 1);
        for (size_t i = 0; i < pk.size(); i++) {
            batch.token[i] = pk.tok[i];
            batch.pos[i] = pk.pos[i];
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = pk.seq[i];
            batch.logits[i] = pk.out[i];
        }
        batch.n_tokens = (int32_t) pk.size();
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        return rc;
    }

    void batch_begin(const common_json & req, std::string & err) {
        reset_cache();
        drop_ckpt();
        cached_key = "\x01none";
        err = apply_cvec(req);
    }

    void batch_end() {
        reset_cache();
        drop_ckpt();
    }

    // {"op":"generate_batch","prompts":[[ids]...],"max_new":N,"stop_ids":[...],
    //  "cvec_b64":...,"il_start":L,"il_end":L}
    // -> {"type":"done_batch","outs":[[ids]...],"stop":"maxlen|cancel|error",
    //     "prefill_ms":F,"decode_ms":F,"error":""}
    // Greedy per sequence; a sequence ends on a stop id (kept, like
    // generate) or at max_new. At most --parallel prompts per call.
    void do_generate_batch(const common_json & req) {
        auto fail = [&](const std::string & e) {
            emit_line(common_json({{"type", "done_batch"},
                                   {"outs", common_json::array()},
                                   {"stop", "error"}, {"prefill_ms", 0.0},
                                   {"decode_ms", 0.0}, {"error", e}}).dump());
        };
        std::vector<std::vector<int>> prompts;
        const auto & parr = req.at("prompts");
        for (size_t i = 0; i < parr.size(); i++) {
            prompts.push_back(parr[i].get<std::vector<int>>());
        }
        int32_t max_new = req.at("max_new").get<int>();
        std::vector<int> stops = req.contains("stop_ids")
            ? req.at("stop_ids").get<std::vector<int>>() : std::vector<int>{};
        const std::unordered_set<int32_t> stop_set(stops.begin(), stops.end());
        const size_t B = prompts.size();
        if (B == 0 || (int32_t) B > n_seq || max_new < 1) {
            fail("generate_batch needs 1..parallel prompts and max_new >= 1");
            return;
        }
        size_t total = 0;
        for (const auto & p : prompts) {
            if (p.empty() || (int32_t) p.size() > n_batch) {
                fail("generate_batch prompt empty or longer than n_batch");
                return;
            }
            total += p.size() + (size_t) max_new;
        }
        if (total > (size_t) n_ctx) {
            fail("generate_batch prompts + max_new exceed n_ctx");
            return;
        }
        std::string err;
        batch_begin(req, err);
        if (!err.empty()) {
            fail(err);
            return;
        }
        int64_t t0 = llama_time_us();
        std::vector<int32_t> next(B, -1);
        // Prefill: pack whole prompts up to n_batch tokens per decode;
        // read each prompt's last-token logits right after its decode.
        Packed pk;
        std::vector<std::pair<size_t, int32_t>> owners; // (seq, batch index)
        auto flush = [&]() -> bool {
            if (pk.size() == 0) {
                return true;
            }
            if (decode_packed(pk) != 0) {
                return false;
            }
            for (auto & o : owners) {
                next[o.first] = argmax(llama_get_logits_ith(ctx, o.second), n_vocab);
            }
            pk.clear();
            owners.clear();
            return true;
        };
        for (size_t i = 0; i < B; i++) {
            if (pk.size() + prompts[i].size() > (size_t) n_batch && !flush()) {
                batch_end();
                fail("llama_decode failed");
                return;
            }
            for (size_t j = 0; j < prompts[i].size(); j++) {
                bool last = j + 1 == prompts[i].size();
                if (last) {
                    owners.push_back({i, (int32_t) pk.size()});
                }
                pk.add(prompts[i][j], (llama_pos) j, (llama_seq_id) i, last);
            }
        }
        if (!flush()) {
            batch_end();
            fail("llama_decode failed");
            return;
        }
        int64_t t1 = llama_time_us();
        std::vector<std::vector<int>> outs(B);
        std::vector<bool> live(B, true);
        std::string stop = "maxlen";
        for (int32_t step = 0; step < max_new; step++) {
            if (cancel_flag.load()) {
                stop = "cancel";
                break;
            }
            for (size_t i = 0; i < B; i++) {
                if (!live[i]) {
                    continue;
                }
                outs[i].push_back(next[i]);
                if (stop_set.count(next[i]) || step + 1 == max_new) {
                    live[i] = false; // stop token kept, never decoded
                    continue;
                }
                owners.push_back({i, (int32_t) pk.size()});
                pk.add(next[i], (llama_pos) (prompts[i].size() + outs[i].size() - 1),
                       (llama_seq_id) i, true);
            }
            if (pk.size() == 0) {
                break;
            }
            if (!flush()) {
                stop = "error";
                err = "llama_decode failed";
                break;
            }
        }
        int64_t t2 = llama_time_us();
        batch_end();
        common_json outs_json = common_json::array();
        for (const auto & o : outs) {
            outs_json.push_back(o);
        }
        emit_line(common_json({{"type", "done_batch"}, {"outs", outs_json},
                               {"stop", stop},
                               {"prefill_ms", (t1 - t0) / 1000.0},
                               {"decode_ms", (t2 - t1) / 1000.0},
                               {"error", stop == "error" ? err : ""}}).dump());
    }

    // {"op":"score_batch","items":[{"ids":[...],"from":K,"last_ids":[...]}],
    //  "cvec_b64":...}
    // -> {"ok":true,"items":[{"lp":[...],"last":[...]}]}
    // lp: log P(ids[j] | ids[:j]) for j in [from, len) (from 0 or absent:
    // none). last: raw last-position logits at last_ids. Items run in
    // groups of at most --parallel sequences, n_batch tokens and 256
    // output rows per decode.
    void do_score_batch(const common_json & req) {
        auto fail = [&](const std::string & e) {
            emit_line(common_json({{"ok", false}, {"error", e}}).dump());
        };
        const auto & items = req.at("items");
        const size_t N = items.size();
        const size_t max_rows = 256;
        std::vector<std::vector<int>> ids(N), last_ids(N);
        std::vector<size_t> from(N, 0);
        auto out_at = [&](size_t k, size_t j) {
            return j + 1 == ids[k].size() ||
                   (from[k] > 0 && j + 1 >= from[k]);
        };
        auto rows_of = [&](size_t k) {
            return (from[k] > 0 ? ids[k].size() - from[k] : 0) + 1;
        };
        for (size_t k = 0; k < N; k++) {
            ids[k] = items[k].at("ids").get<std::vector<int>>();
            if (items[k].contains("from")) {
                from[k] = (size_t) std::max(0, items[k].at("from").get<int>());
            }
            if (items[k].contains("last_ids")) {
                last_ids[k] = items[k].at("last_ids").get<std::vector<int>>();
            }
            if (ids[k].empty() || ids[k].size() > (size_t) n_batch ||
                ids[k].size() > (size_t) n_ctx || from[k] >= ids[k].size() ||
                rows_of(k) > max_rows) {
                fail("score_batch item empty, too long, or bad from");
                return;
            }
            for (int t : last_ids[k]) {
                if (t < 0 || t >= n_vocab) {
                    fail("score_batch last_ids out of vocab");
                    return;
                }
            }
        }
        std::string err;
        batch_begin(req, err);
        if (!err.empty()) {
            fail(err);
            return;
        }
        std::vector<std::vector<float>> lp(N), last(N);
        size_t k0 = 0;
        while (k0 < N) {
            if (cancel_flag.load()) {
                batch_end();
                emit_line(common_json({{"ok", false}, {"cancelled", true},
                                       {"error", "cancelled"}}).dump());
                return;
            }
            // One group: at most n_seq items, n_batch tokens, max_rows rows.
            size_t k1 = k0, toks = 0, rows = 0;
            while (k1 < N && (int32_t) (k1 - k0) < n_seq) {
                if (k1 > k0 && (toks + ids[k1].size() > (size_t) n_batch ||
                                toks + ids[k1].size() > (size_t) n_ctx ||
                                rows + rows_of(k1) > max_rows)) {
                    break;
                }
                toks += ids[k1].size();
                rows += rows_of(k1);
                k1++;
            }
            Packed pk;
            std::vector<std::pair<size_t, size_t>> where; // (item, position)
            std::vector<int32_t> at;                      // batch index
            for (size_t k = k0; k < k1; k++) {
                for (size_t j = 0; j < ids[k].size(); j++) {
                    bool o = out_at(k, j);
                    if (o) {
                        where.push_back({k, j});
                        at.push_back((int32_t) pk.size());
                    }
                    pk.add(ids[k][j], (llama_pos) j, (llama_seq_id) (k - k0), o);
                }
            }
            if (decode_packed(pk) != 0) {
                batch_end();
                fail("llama_decode failed");
                return;
            }
            // Log softmax over the vocab per row is CPU work: spread the
            // rows over threads, then append in row order.
            std::vector<float> row_lp(where.size(), 0.0f);
            std::vector<const float *> row_lg(where.size());
            for (size_t r = 0; r < where.size(); r++) {
                row_lg[r] = llama_get_logits_ith(ctx, at[r]);
            }
            size_t n_thr = std::max<size_t>(1, std::min<size_t>(
                std::thread::hardware_concurrency(), where.size()));
            std::vector<std::thread> pool;
            for (size_t w = 0; w < n_thr; w++) {
                pool.emplace_back([&, w] {
                    for (size_t r = w; r < where.size(); r += n_thr) {
                        size_t k = where[r].first, j = where[r].second;
                        if (j + 1 < ids[k].size()) {
                            row_lp[r] = log_softmax_at(row_lg[r], ids[k][j + 1]);
                        }
                    }
                });
            }
            for (auto & t : pool) {
                t.join();
            }
            for (size_t r = 0; r < where.size(); r++) {
                size_t k = where[r].first, j = where[r].second;
                if (j + 1 == ids[k].size()) {
                    for (int t : last_ids[k]) {
                        last[k].push_back(row_lg[r][t]);
                    }
                } else {
                    lp[k].push_back(row_lp[r]);
                }
            }
            llama_memory_clear(llama_get_memory(ctx), false);
            k0 = k1;
        }
        batch_end();
        common_json outs = common_json::array();
        for (size_t k = 0; k < N; k++) {
            outs.push_back(common_json({{"lp", lp[k]}, {"last", last[k]}}));
        }
        emit_line(common_json({{"ok", true}, {"items", outs}}).dump());
    }

    // log softmax(lg)[t] (double accumulation, max-shifted).
    float log_softmax_at(const float * lg, int32_t t) const {
        float mx = lg[0];
        for (int32_t i = 1; i < n_vocab; i++) {
            mx = std::max(mx, lg[i]);
        }
        double sum = 0.0;
        for (int32_t i = 0; i < n_vocab; i++) {
            sum += std::exp((double) lg[i] - mx);
        }
        return (float) ((double) lg[t] - mx - std::log(sum));
    }
};

struct Request {
    bool eof = false;
    std::string line;
};

int main(int argc, char ** argv) {
    llama_log_set(log_to_stderr, nullptr);

    std::string model_path;
    int32_t ctx_size = 8192;
    int32_t ubatch = 512;
    int32_t threads = 0; // 0 = llama.cpp default
    int32_t parallel = 1; // max sequences per batch op
    bool prefix_cache = true;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-m") {
            model_path = need("-m");
        } else if (a == "--ctx-size") {
            ctx_size = std::stoi(need("--ctx-size"));
        } else if (a == "--ubatch") {
            ubatch = std::stoi(need("--ubatch"));
        } else if (a == "--threads") {
            threads = std::stoi(need("--threads"));
        } else if (a == "--parallel") {
            parallel = std::max(1, std::stoi(need("--parallel")));
        } else if (a == "--no-prefix-cache") {
            prefix_cache = false;
        } else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: lobot-llamad -m model.gguf "
                             "[--ctx-size N] [--ubatch N] [--threads N] "
                             "[--no-prefix-cache] [--parallel N]\n");
        return 2;
    }

    Daemon daemon;
    if (!daemon.load(model_path, ctx_size, ubatch, threads, parallel)) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    daemon.prefix_cache = prefix_cache;
    daemon.use_ckpt = llama_model_is_recurrent(daemon.model) ||
                      llama_model_is_hybrid(daemon.model);
    // Ready banner on stderr only: stdout stays protocol-clean.
    std::fprintf(stderr, "lobot-llamad ready: %s\n", model_path.c_str());
    std::fflush(stderr);

    std::queue<Request> queue;
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    bool reader_done = false;

    // Reader thread owns stdin so "cancel" lands mid-generate.
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            bool is_cancel = false;
            try {
                common_json req = common_json::parse(line);
                if (req.contains("op") &&
                    req.at("op").get<std::string>() == "cancel") {
                    is_cancel = true;
                }
            } catch (const std::exception &) {
                // Worker reports the parse error.
            }
            if (is_cancel) {
                daemon.cancel_flag.store(true);
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(queue_mu);
                // Reset here, not in the worker: a cancel read after
                // this line targets this request and must survive
                // until the worker starts it. The client sends the
                // next request only after the previous one is done.
                daemon.cancel_flag.store(false);
                queue.push(Request{false, line});
            }
            queue_cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lock(queue_mu);
            queue.push(Request{true, ""});
            reader_done = true;
        }
        queue_cv.notify_one();
    });

    while (true) {
        Request req;
        {
            std::unique_lock<std::mutex> lock(queue_mu);
            queue_cv.wait(lock, [&] { return !queue.empty(); });
            req = queue.front();
            queue.pop();
        }
        if (req.eof) {
            break;
        }
        std::string op;
        try {
            common_json parsed = common_json::parse(req.line);
            op = parsed.contains("op") ? parsed.at("op").get<std::string>() : "";
            if (op == "info") {
                daemon.do_info();
            } else if (op == "prefill") {
                daemon.do_prefill(parsed);
            } else if (op == "generate_batch") {
                daemon.do_generate_batch(parsed);
            } else if (op == "score_batch") {
                daemon.do_score_batch(parsed);
            } else if (op == "generate") {
                daemon.do_generate(parsed);
            } else if (op == "shutdown") {
                emit_line(common_json({{"ok", true}}).dump());
                break;
            } else {
                emit_line(common_json({{"ok", false},
                                       {"error", "unknown op"}}).dump());
            }
        } catch (const std::exception & e) {
            std::string msg = std::string("bad request: ") + e.what();
            if (op == "generate") {
                emit_line(common_json({{"type", "done"}, {"ids", std::vector<int>{}},
                                       {"stop", "error"}, {"cancelled", false},
                                       {"prefill_ms", 0.0}, {"decode_ms", 0.0},
                                       {"n_prefill", 0}, {"error", msg}}).dump());
            } else {
                emit_line(common_json({{"ok", false}, {"error", msg}}).dump());
            }
        }
    }

    if (reader.joinable()) {
        // stdin EOF already consumed, or shutdown: detach-safe join
        // would block on getline, so detach and let exit clean up.
        reader.detach();
    }
    return 0;
}
