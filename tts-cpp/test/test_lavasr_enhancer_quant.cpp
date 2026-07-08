// Self-contained round-trip test for the LavaSR enhancer QUANTIZED GGUF path
// (Q4_0 / Q8_0 "dequant-at-load", QVAC-21906).
//
// scripts/convert-lavasr-enhancer-to-gguf.py --ftype q4_0|q8_0 (and
// scripts/requantize-gguf.py) block-quantize only the big 2-D matmul weights —
// the 8x ConvNeXt pwconv1/pwconv2 and the spec-head Linear — and keep the K=7
// conv kernels + LayerNorm scales + biases + gamma at F16/F32. load_enhancer_gguf()
// then dequantizes every tensor to F32 at load (ggml_get_type_traits()->to_float),
// so the scalar/graph forward math is unchanged and the only cost is the
// quantization error baked into the stored weights.
//
// This test builds a small enhancer with deterministic pseudo-random weights,
// writes it to a temp GGUF three ways (F32 baseline, Q8_0, Q4_0 — quantizing
// exactly the pwconv/spec-head tensors), loads each back through the real
// load_enhancer_gguf() + scalar enhancer_spec_forward(), and asserts:
//   * the quantized GGUFs LOAD (exercise the ggml_is_quantized dequant branch),
//   * their spectral output is finite and well-correlated with the F32 baseline
//     (a byte-reinterpret / wrong-dtype bug tanks cos-sim to ~0 or NaN),
//   * Q8_0 is near-lossless and at least as faithful as Q4_0.
// No model download, no ONNX fixtures — always runs in CI.

#include "lavasr/enhancer_core.h"
#include "lavasr/enhancer_gguf.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using tts_cpp::lavasr::EnhancerWeights;
using tts_cpp::lavasr::EnhTensor;

static void add_tensor(EnhancerWeights & w, const std::string & name, int64_t n,
                       std::mt19937 & rng, float mean, float stddev) {
    std::normal_distribution<float> nd(mean, stddev);
    EnhTensor                       t;
    t.data.resize(static_cast<size_t>(n));
    for (auto & v : t.data) {
        v = nd(rng);
    }
    w.t[name] = std::move(t);
}

// Deterministic pseudo-random weights sized to w's dims (mirrors the sizing in
// test_lavasr_enhancer_ggml.cpp so the two self-tests stay consistent).
static void fill_random_weights(EnhancerWeights & w, std::mt19937 & rng) {
    const int C = w.dim, F = w.ffn_dim, M = w.n_mels, K = w.kernel, B = w.spec_bins;
    add_tensor(w, "enhancer.embed.weight", static_cast<int64_t>(C) * M * K, rng, 0.0f, 0.05f);
    add_tensor(w, "enhancer.embed.bias", C, rng, 0.0f, 0.02f);
    add_tensor(w, "enhancer.norm.weight", C, rng, 1.0f, 0.05f);
    add_tensor(w, "enhancer.norm.bias", C, rng, 0.0f, 0.02f);
    for (int i = 0; i < w.n_blocks; i++) {
        const std::string p = "enhancer.block." + std::to_string(i) + ".";
        add_tensor(w, p + "dwconv.weight", static_cast<int64_t>(C) * K, rng, 0.0f, 0.1f);
        add_tensor(w, p + "dwconv.bias", C, rng, 0.0f, 0.02f);
        add_tensor(w, p + "norm.weight", C, rng, 1.0f, 0.05f);
        add_tensor(w, p + "norm.bias", C, rng, 0.0f, 0.02f);
        add_tensor(w, p + "pwconv1.weight", static_cast<int64_t>(F) * C, rng, 0.0f, 0.05f);
        add_tensor(w, p + "pwconv1.bias", F, rng, 0.0f, 0.02f);
        add_tensor(w, p + "pwconv2.weight", static_cast<int64_t>(C) * F, rng, 0.0f, 0.05f);
        add_tensor(w, p + "pwconv2.bias", C, rng, 0.0f, 0.02f);
        add_tensor(w, p + "gamma", C, rng, 0.1f, 0.02f);
    }
    add_tensor(w, "enhancer.final_norm.weight", C, rng, 1.0f, 0.05f);
    add_tensor(w, "enhancer.final_norm.bias", C, rng, 0.0f, 0.02f);
    add_tensor(w, "spec_head.out.weight", static_cast<int64_t>(2) * B * C, rng, 0.0f, 0.05f);
    add_tensor(w, "spec_head.out.bias", 2 * B, rng, 0.0f, 0.02f);
}

// The three tensors the shared quant policy (should_quantize) block-quantizes:
// the pointwise ConvNeXt Linears and the spec-head Linear.  Everything else
// (K=7 conv kernels, norms, biases, gamma) stays F32.
static bool is_quantized_weight(const std::string & name) {
    auto ends = [&](const char * s) {
        const size_t ls = std::strlen(s);
        return name.size() >= ls && name.compare(name.size() - ls, ls, s) == 0;
    };
    return ends(".pwconv1.weight") || ends(".pwconv2.weight") ||
           name == "spec_head.out.weight";
}

// Write `w` to a temp enhancer GGUF matching convert-lavasr-enhancer-to-gguf.py's
// schema.  When `qtype != GGML_TYPE_F32`, the pwconv/spec-head weights are stored
// block-quantized (via ggml_quantize_chunk along the reduction dim ne0); all
// other tensors stay F32.  Returns the path, or "" on failure.
static std::string write_enhancer_gguf(const EnhancerWeights & w, enum ggml_type qtype,
                                       const char * tag) {
    const int C = w.dim, F = w.ffn_dim, M = w.n_mels, K = w.kernel, B = w.spec_bins;

    struct Entry {
        std::string          name;
        std::vector<int64_t> ne; // ggml order
    };
    std::vector<Entry> roster;
    roster.push_back({"enhancer.embed.weight", {K, M, C}});
    roster.push_back({"enhancer.embed.bias", {C}});
    roster.push_back({"enhancer.norm.weight", {C}});
    roster.push_back({"enhancer.norm.bias", {C}});
    for (int i = 0; i < w.n_blocks; i++) {
        const std::string p = "enhancer.block." + std::to_string(i) + ".";
        roster.push_back({p + "dwconv.weight", {K, 1, C}});
        roster.push_back({p + "dwconv.bias", {C}});
        roster.push_back({p + "norm.weight", {C}});
        roster.push_back({p + "norm.bias", {C}});
        roster.push_back({p + "pwconv1.weight", {C, F}});
        roster.push_back({p + "pwconv1.bias", {F}});
        roster.push_back({p + "pwconv2.weight", {F, C}});
        roster.push_back({p + "pwconv2.bias", {C}});
        roster.push_back({p + "gamma", {C}});
    }
    roster.push_back({"enhancer.final_norm.weight", {C}});
    roster.push_back({"enhancer.final_norm.bias", {C}});
    roster.push_back({"spec_head.out.weight", {C, 2 * B}});
    roster.push_back({"spec_head.out.bias", {2 * B}});

    // Size the ctx for the F32 upper bound (quantized tensors need less).
    size_t bytes = 0;
    for (const Entry & e : roster) {
        int64_t n = 1;
        for (int64_t d : e.ne) {
            n *= d;
        }
        bytes += static_cast<size_t>(n) * sizeof(float);
    }
    ggml_init_params ip = {
        bytes + (roster.size() + 1) * ggml_tensor_overhead() + 64 * roster.size(), nullptr,
        /*no_alloc=*/false};
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: ggml_init for GGUF writer failed\n");
        return std::string();
    }

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "lavasr-enhancer");
    gguf_set_val_u32(g, "lavasr.enhancer.dim", static_cast<uint32_t>(w.dim));
    gguf_set_val_u32(g, "lavasr.enhancer.ffn_dim", static_cast<uint32_t>(w.ffn_dim));
    gguf_set_val_u32(g, "lavasr.enhancer.n_blocks", static_cast<uint32_t>(w.n_blocks));
    gguf_set_val_u32(g, "lavasr.enhancer.n_mels", static_cast<uint32_t>(w.n_mels));
    gguf_set_val_u32(g, "lavasr.enhancer.kernel", static_cast<uint32_t>(w.kernel));
    gguf_set_val_u32(g, "lavasr.enhancer.n_fft", static_cast<uint32_t>(w.n_fft));
    gguf_set_val_u32(g, "lavasr.enhancer.hop", static_cast<uint32_t>(w.hop));
    gguf_set_val_u32(g, "lavasr.enhancer.win", static_cast<uint32_t>(w.win));
    gguf_set_val_u32(g, "lavasr.enhancer.spec_bins", static_cast<uint32_t>(w.spec_bins));
    gguf_set_val_f32(g, "lavasr.enhancer.clip_max", w.clip_max);
    gguf_set_val_f32(g, "lavasr.enhancer.layernorm_eps", w.ln_eps);
    gguf_set_val_u32(g, "lavasr.enhancer.work_sample_rate",
                     static_cast<uint32_t>(w.work_sample_rate));
    gguf_set_val_u32(g, "lavasr.enhancer.mel_ref_sample_rate",
                     static_cast<uint32_t>(w.mel_ref_sample_rate));

    for (const Entry & e : roster) {
        const std::vector<float> & src = w.get(e.name).data;
        const bool quant = qtype != GGML_TYPE_F32 && is_quantized_weight(e.name);
        if (quant) {
            // Quantize along ne0 (the reduction dim): nrows rows of n_per_row.
            const int64_t n_per_row = e.ne[0];
            int64_t       nrows     = 1;
            for (size_t d = 1; d < e.ne.size(); d++) {
                nrows *= e.ne[d];
            }
            ggml_tensor * t = ggml_new_tensor(ctx, qtype, static_cast<int>(e.ne.size()),
                                              e.ne.data());
            ggml_set_name(t, e.name.c_str());
            ggml_quantize_chunk(qtype, src.data(), t->data, /*start=*/0, nrows, n_per_row,
                                /*imatrix=*/nullptr);
            gguf_add_tensor(g, t);
        } else {
            ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32,
                                              static_cast<int>(e.ne.size()), e.ne.data());
            ggml_set_name(t, e.name.c_str());
            std::memcpy(t->data, src.data(), src.size() * sizeof(float));
            gguf_add_tensor(g, t);
        }
    }

    const char * tmpdir = std::getenv("TMPDIR");
    std::string  path    = std::string(tmpdir ? tmpdir : "/tmp") +
                       "/test-lavasr-enhancer-" + tag + ".gguf";
    const bool ok = gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) {
        std::fprintf(stderr, "FAIL: could not write enhancer GGUF to %s\n", path.c_str());
        return std::string();
    }
    return path;
}

struct Diff {
    double cos = 0.0;
    float  max_abs = 0.0f;
    bool   finite = true;
};

static Diff compare(const std::vector<float> & a, const std::vector<float> & b) {
    Diff   d;
    double dot = 0, na = 0, nb = 0;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            d.finite = false;
        }
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
        d.max_abs = std::max(d.max_abs, std::fabs(a[i] - b[i]));
    }
    d.cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    return d;
}

// Load a GGUF, run the scalar spec forward on `mel`, return real/imag.
static bool forward(const std::string & path, const std::vector<float> & mel, int T,
                    std::vector<float> & re, std::vector<float> & im) {
    EnhancerWeights w;
    std::string     err;
    if (!tts_cpp::lavasr::load_enhancer_gguf(path, w, &err)) {
        std::fprintf(stderr, "FAIL: load_enhancer_gguf(%s): %s\n", path.c_str(), err.c_str());
        return false;
    }
    tts_cpp::lavasr::enhancer_spec_forward(w, mel, T, re, im);
    return true;
}

int main() {
    // Small but structurally complete enhancer; dims are multiples of the 32-
    // element Q4_0/Q8_0 block so ggml_quantize_chunk accepts the reduction dim.
    EnhancerWeights w;
    w.dim       = 64;
    w.ffn_dim   = 128;
    w.n_blocks  = 3;
    w.n_mels    = 32;
    w.kernel    = 7;
    w.spec_bins = 40;
    w.clip_max  = 1000.0f;
    w.ln_eps    = 1e-6f;

    std::mt19937 rng(0xC0FFEEu);
    fill_random_weights(w, rng);

    const int          M = w.n_mels, T = 60;
    std::vector<float> mel(static_cast<size_t>(M) * T);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : mel) {
        v = nd(rng);
    }

    const std::string p_f32 = write_enhancer_gguf(w, GGML_TYPE_F32, "f32");
    const std::string p_q8  = write_enhancer_gguf(w, GGML_TYPE_Q8_0, "q8_0");
    const std::string p_q4  = write_enhancer_gguf(w, GGML_TYPE_Q4_0, "q4_0");
    if (p_f32.empty() || p_q8.empty() || p_q4.empty()) {
        return 1;
    }

    std::printf("LavaSR enhancer quant round-trip (C=%d F=%d blocks=%d M=%d B=%d T=%d):\n",
                w.dim, w.ffn_dim, w.n_blocks, M, w.spec_bins, T);

    int failures = 0;
    std::vector<float> re0, im0;
    if (!forward(p_f32, mel, T, re0, im0)) {
        ++failures;
    }

    struct Case { const char * name; std::string path; double cos_min; };
    Case cases[] = {
        {"q8_0", p_q8, 0.99},  // near-lossless
        {"q4_0", p_q4, 0.80},  // aggressive but must stay well-correlated (not garbage)
    };

    double q8_cos = 1.0, q4_cos = 1.0;
    for (Case & c : cases) {
        std::vector<float> re, im;
        if (!forward(c.path, mel, T, re, im)) { ++failures; continue; }
        Diff dr = compare(re, re0);
        Diff di = compare(im, im0);
        std::printf("  %-5s real: cos=%.6f max_abs=%.3e | imag: cos=%.6f  (min cos required %.2f)\n",
                    c.name, dr.cos, dr.max_abs, di.cos, c.cos_min);
        if (!dr.finite || !di.finite) {
            std::fprintf(stderr, "FAIL[%s]: non-finite output (dequant produced garbage)\n", c.name);
            ++failures;
        }
        if (dr.cos < c.cos_min || di.cos < c.cos_min) {
            std::fprintf(stderr, "FAIL[%s]: cos below %.2f (real=%.6f imag=%.6f)\n",
                         c.name, c.cos_min, dr.cos, di.cos);
            ++failures;
        }
        if (std::strcmp(c.name, "q8_0") == 0) { q8_cos = std::min(dr.cos, di.cos); }
        if (std::strcmp(c.name, "q4_0") == 0) { q4_cos = std::min(dr.cos, di.cos); }
    }

    // Q8_0 (8.5 bits) must be at least as faithful as Q4_0 (4.5 bits) — a
    // reversed ordering would signal the dtype branch mixed the two up.
    if (q8_cos < q4_cos) {
        std::fprintf(stderr, "FAIL: Q8_0 cos (%.6f) < Q4_0 cos (%.6f) — unexpected ordering\n",
                     q8_cos, q4_cos);
        ++failures;
    }

    std::remove(p_f32.c_str());
    std::remove(p_q8.c_str());
    std::remove(p_q4.c_str());

    if (failures == 0) {
        std::printf("OK: quantized enhancer GGUFs dequantize + run (Q8_0 near-lossless, Q4_0 correlated)\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d check(s)\n", failures);
    return 1;
}
