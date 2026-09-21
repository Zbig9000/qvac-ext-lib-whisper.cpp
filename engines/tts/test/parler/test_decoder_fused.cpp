// Model-free decoder coverage for the CPU decode optimizations: fused q|k|v +
// stacked LM-head projections must reproduce the separate-weight logits
// byte-for-byte on the decode step (and within float tolerance on prefill,
// where the wider GEMM may tile differently), and on a host backend the
// prefill/step logits hand out a zero-copy view into the graph buffer.
// Runs on a synthetic two-layer model, so it needs no GGUF fixture.

#include "parler/internal.h"
#include "backend_selection.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace tts_cpp::parler::detail;

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);   \
        }                                                                      \
    } while (0)

namespace {

constexpr int D          = 64;
constexpr int N_HEAD     = 4;
constexpr int N_LAYER    = 2;
constexpr int D_FF       = 128;
constexpr int N_CB       = 3;
constexpr int VOCAB      = 32;
constexpr int CROSS_LEN  = 5;
constexpr int N_CTX      = 64;
constexpr int MAX_POS    = 64;
constexpr int BOS_ID     = 30;
constexpr int PROMPT_VOCAB = 8;
constexpr int N_THREADS  = 2;
constexpr int N_STEPS    = 3;

// xorshift-based deterministic fill: the values only need to be finite,
// spread out, and identical across the two runs.
struct det_rng {
    uint32_t state = 0x9E3779B9u;
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return ((float) (state & 0xFFFFFF) / (float) 0xFFFFFF - 0.5f) * 0.2f;
    }
};

void fill_tensor(ggml_tensor * t, det_rng & rng, float bias) {
    std::vector<float> host(ggml_nelements(t));
    for (float & v : host) v = bias + rng.next();
    ggml_backend_tensor_set(t, host.data(), 0, host.size() * sizeof(float));
}

ggml_tensor * new_weight(ggml_context * ctx, int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
}

bool build_model(parler_model & model) {
    parler_hparams & hp = model.hparams;
    hp.dec_n_layer    = N_LAYER;
    hp.dec_d_model    = D;
    hp.dec_n_head     = N_HEAD;
    hp.dec_d_ff       = D_FF;
    hp.n_codebooks    = N_CB;
    hp.dec_vocab      = VOCAB;
    hp.dec_ln_eps     = 1e-5f;
    hp.bos_id         = BOS_ID;
    hp.max_position   = MAX_POS;
    hp.n_ctx          = N_CTX;
    hp.gen_max_length = 8;

    ::tts_cpp::detail::ensure_backends_loaded();
    model.backend = ::tts_cpp::detail::init_cpu_backend();
    if (!model.backend) return false;
    model.on_gpu  = false;
    model.use_fa  = false;
    model.kv_type = GGML_TYPE_F32;

    {
        ggml_init_params ip = { 64 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
        model.ctx_w = ggml_init(ip);
        if (!model.ctx_w) return false;
    }
    model.dec_layers.resize(N_LAYER);
    for (auto & l : model.dec_layers) {
        l.attn_norm_w  = new_weight(model.ctx_w, D, 1);
        l.attn_norm_b  = new_weight(model.ctx_w, D, 1);
        l.q            = new_weight(model.ctx_w, D, D);
        l.k            = new_weight(model.ctx_w, D, D);
        l.v            = new_weight(model.ctx_w, D, D);
        l.o            = new_weight(model.ctx_w, D, D);
        l.cross_norm_w = new_weight(model.ctx_w, D, 1);
        l.cross_norm_b = new_weight(model.ctx_w, D, 1);
        l.cq           = new_weight(model.ctx_w, D, D);
        l.co           = new_weight(model.ctx_w, D, D);
        l.ffn_norm_w   = new_weight(model.ctx_w, D, 1);
        l.ffn_norm_b   = new_weight(model.ctx_w, D, 1);
        l.up           = new_weight(model.ctx_w, D, D_FF);
        l.down         = new_weight(model.ctx_w, D_FF, D);
    }
    model.embed_positions   = new_weight(model.ctx_w, D, MAX_POS);
    model.embed_prompts     = new_weight(model.ctx_w, D, PROMPT_VOCAB);
    model.dec_output_norm_w = new_weight(model.ctx_w, D, 1);
    model.dec_output_norm_b = new_weight(model.ctx_w, D, 1);
    model.dec_embed.resize(N_CB);
    model.lm_heads.resize(N_CB);
    for (int k = 0; k < N_CB; ++k) {
        model.dec_embed[k] = new_weight(model.ctx_w, D, BOS_ID + 2);
        model.lm_heads[k]  = new_weight(model.ctx_w, D, VOCAB);
    }
    model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buffer_w) return false;

    det_rng rng;
    for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_w); t;
         t = ggml_get_next_tensor(model.ctx_w, t)) {
        fill_tensor(t, rng, t->ne[1] == 1 ? 1.0f : 0.0f);
    }

    {
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
        model.ctx_kv = ggml_init(ip);
        if (!model.ctx_kv) return false;
        const int64_t rows = (int64_t) N_CTX * N_LAYER;
        model.memory_k = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, D, rows);
        model.memory_v = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, D, rows);
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
        if (!model.buffer_kv) return false;
    }

    {
        ggml_init_params ip = { (size_t) (2 * N_LAYER + 2) * ggml_tensor_overhead(),
                                nullptr, /*no_alloc=*/ true };
        model.ctx_cross = ggml_init(ip);
        if (!model.ctx_cross) return false;
        model.cross_k.resize(N_LAYER);
        model.cross_v_t.resize(N_LAYER);
        for (int l = 0; l < N_LAYER; ++l) {
            model.cross_k[l]   = new_weight(model.ctx_cross, D, CROSS_LEN);
            model.cross_v_t[l] = new_weight(model.ctx_cross, CROSS_LEN, D);
        }
        model.buffer_cross = ggml_backend_alloc_ctx_tensors(model.ctx_cross, model.backend);
        if (!model.buffer_cross) return false;
        for (int l = 0; l < N_LAYER; ++l) {
            fill_tensor(model.cross_k[l], rng, 0.0f);
            fill_tensor(model.cross_v_t[l], rng, 0.0f);
        }
        model.cross_len = CROSS_LEN;
    }
    return true;
}

// prefill + N_STEPS teacher-forced steps; returns one logits block per call.
bool run_decode(const parler_model & model, ggml_gallocr_t allocr,
                std::vector<std::vector<float>> & out, bool & zero_copy) {
    const size_t n = (size_t) VOCAB * N_CB;
    out.clear();
    zero_copy = true;

    const std::vector<int32_t> prompt_ids  = { 0, 1, 2 };
    const std::vector<int32_t> start_frame((size_t) N_CB, BOS_ID);
    parler_step_logits logits;
    int n_past = 0;
    if (!parler_dec_prefill(model, prompt_ids, start_frame, allocr, N_THREADS,
                            logits, n_past)) return false;
    zero_copy = zero_copy && logits.copy.empty();
    out.emplace_back(logits.view, logits.view + n);

    for (int s = 0; s < N_STEPS; ++s) {
        const std::vector<int32_t> frame = { (int32_t) (s * 3 + 1) % VOCAB,
                                             (int32_t) (s * 3 + 2) % VOCAB,
                                             (int32_t) (s * 3 + 3) % VOCAB };
        if (!parler_dec_step(model, frame, n_past, allocr, N_THREADS, logits)) return false;
        n_past++;
        zero_copy = zero_copy && logits.copy.empty();
        out.emplace_back(logits.view, logits.view + n);
    }
    return true;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

} // namespace

int main() {
    parler_model model;
    if (!build_model(model)) {
        fprintf(stderr, "parler decoder fused: model construction failed\n");
        return 1;
    }
    ggml_gallocr_t allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) {
        fprintf(stderr, "parler decoder fused: allocator creation failed\n");
        return 1;
    }

    std::vector<std::vector<float>> separate, fused;
    bool zero_copy = false;
    CHECK(run_decode(model, allocr, separate, zero_copy), "separate-weight decode runs");
    CHECK(zero_copy, "host backend hands out zero-copy logits views");

    CHECK(parler_fuse_decode_weights(model), "fusing the decode weights succeeds");
    bool wired = model.lm_head_stacked != nullptr;
    for (const auto & l : model.dec_layers) wired = wired && l.qkv != nullptr;
    CHECK(wired, "fusion wires qkv on every layer and the stacked LM head");

    CHECK(run_decode(model, allocr, fused, zero_copy), "fused-weight decode runs");
    if (separate.size() == fused.size() && !separate.empty()) {
        // prefill (N > 1) may tile the wider GEMM differently; the N=1 decode
        // steps are per-row dot products and must not move a single bit
        CHECK(max_abs_diff(separate[0], fused[0]) <= 1e-6f,
              "fused prefill logits match within float tolerance");
        for (size_t s = 1; s < separate.size(); ++s) {
            CHECK(std::memcmp(separate[s].data(), fused[s].data(),
                              separate[s].size() * sizeof(float)) == 0,
                  "fused step logits are byte-identical to separate weights");
        }
    } else {
        CHECK(false, "both decodes produced the same number of logits blocks");
    }

    ggml_gallocr_free(allocr);
    parler_free_model(model);

    if (g_failures == 0) {
        fprintf(stderr, "parler decoder fused: PASS\n");
        return 0;
    }
    fprintf(stderr, "parler decoder fused: %d failure(s)\n", g_failures);
    return 1;
}
