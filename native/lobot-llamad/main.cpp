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
//         "n_prefill":N,"error":null|"..."}.
//        Greedy (argmax). Stops only on stop_ids. cancel works between tokens.
//   {"op":"prefill","ids":[...],"cvec_b64":"..."|null,"il_start":L,"il_end":L}
//     -> {"ok":true,"logits_b64":"..."} (full last-position logits, f32 LE)
//   {"op":"cancel"} -> {"ok":true} (flags the in-flight request)
//   {"op":"shutdown"} -> exits 0. stdin EOF also exits 0.
//
// Every request starts from a clean slate: KV cleared, cvec set or
// cleared, nothing carried over. cvec_b64 is a full n_layer*n_embd
// float buffer (slot L-1 per 0-based layer L); il_start/il_end are
// 0-based touched layers.

#include "llama.h"

#include "base64.hpp"
#include "json.h"

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
    std::atomic<bool> cancel_flag{false};

    bool load(const std::string & path, int32_t ctx_size, int32_t ubatch, int32_t threads) {
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

    void do_prefill(const common_json & req) {
        cancel_flag.store(false);
        llama_memory_clear(llama_get_memory(ctx), true);
        std::string err = apply_cvec(req);
        if (!err.empty()) {
            emit_line(common_json({{"ok", false}, {"error", err}}).dump());
            return;
        }
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
        int32_t n_past = 0;
        for (size_t off = 0; off < toks.size(); off += (size_t) n_ubatch) {
            size_t n = std::min((size_t) n_ubatch, toks.size() - off);
            std::vector<llama_token> chunk(toks.begin() + off,
                                           toks.begin() + off + n);
            bool last = off + n == toks.size();
            if (decode_chunk(chunk, n_past, last) != 0) {
                emit_line(common_json({{"ok", false},
                                       {"error", "llama_decode failed"}}).dump());
                return;
            }
            n_past += (int32_t) n;
        }
        const float * logits = llama_get_logits(ctx);
        std::string blob((const char *) logits,
                         (size_t) n_vocab * sizeof(float));
        emit_line(common_json({{"ok", true},
                               {"logits_b64", base64::encode(blob)}}).dump());
    }

    void do_generate(const common_json & req) {
        cancel_flag.store(false);
        llama_memory_clear(llama_get_memory(ctx), true);
        std::string err = apply_cvec(req);
        if (!err.empty()) {
            emit_line(common_json({{"type", "done"}, {"ids", std::vector<int>{}},
                                   {"stop", "error"}, {"cancelled", false},
                                   {"prefill_ms", 0.0}, {"decode_ms", 0.0},
                                   {"n_prefill", 0}, {"error", err}}).dump());
            return;
        }
        std::vector<int> ids = req.at("ids").get<std::vector<int>>();
        int32_t max_new = req.at("max_new").get<int>();
        std::vector<int> stops =
            req.contains("stop_ids") ? req.at("stop_ids").get<std::vector<int>>()
                                     : std::vector<int>{};
        const std::unordered_set<int32_t> stop_set(stops.begin(), stops.end());
        if (ids.empty() || max_new < 1) {
            emit_line(common_json({{"type", "done"}, {"ids", std::vector<int>{}},
                                   {"stop", "error"}, {"cancelled", false},
                                   {"prefill_ms", 0.0}, {"decode_ms", 0.0},
                                   {"n_prefill", 0},
                                   {"error", "generate needs ids and max_new >= 1"}}).dump());
            return;
        }
        if ((int32_t) ids.size() + max_new > n_ctx) {
            emit_line(common_json({{"type", "done"}, {"ids", std::vector<int>{}},
                                   {"stop", "error"}, {"cancelled", false},
                                   {"prefill_ms", 0.0}, {"decode_ms", 0.0},
                                   {"n_prefill", 0},
                                   {"error", "prompt + max_new exceeds n_ctx"}}).dump());
            return;
        }
        std::vector<llama_token> toks(ids.begin(), ids.end());
        llama_perf_context_reset(ctx);
        int64_t t0 = llama_time_us();
        int32_t n_past = 0;
        for (size_t off = 0; off < toks.size(); off += (size_t) n_ubatch) {
            size_t n = std::min((size_t) n_ubatch, toks.size() - off);
            std::vector<llama_token> chunk(toks.begin() + off,
                                           toks.begin() + off + n);
            bool last = off + n == toks.size();
            if (decode_chunk(chunk, n_past, last) != 0) {
                emit_line(common_json({{"type", "done"}, {"ids", std::vector<int>{}},
                                       {"stop", "error"}, {"cancelled", false},
                                       {"prefill_ms", 0.0}, {"decode_ms", 0.0},
                                       {"n_prefill", 0},
                                       {"error", "llama_decode failed"}}).dump());
                return;
            }
            n_past += (int32_t) n;
        }
        int64_t t1 = llama_time_us();
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
                // needs the same ids on both sides.
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
                break;
            }
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
                               {"n_prefill", n_past - (int32_t) out.size()},
                               {"error", stop == "error" ? err : ""}}).dump());
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
        } else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: lobot-llamad -m model.gguf "
                             "[--ctx-size N] [--ubatch N] [--threads N]\n");
        return 2;
    }

    Daemon daemon;
    if (!daemon.load(model_path, ctx_size, ubatch, threads)) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
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
                emit_line(common_json({{"ok", true}}).dump());
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(queue_mu);
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
