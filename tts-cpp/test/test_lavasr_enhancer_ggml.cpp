// Parity test for the LavaSR enhancer GGML/GPU forward
// (enhancer_ggml_spec_forward) against the scalar CPU core
// (enhancer_spec_forward), which is itself validated bit-comparable to
// onnxruntime by test-lavasr-enhancer-core.  Since scalar == onnx golden and
// this asserts ggml == scalar, it transitively validates the GPU port against
// the reference network.
//
// Primary path is SELF-CONTAINED: it builds a small enhancer with deterministic
// pseudo-random weights (no model download / fixtures), so it always runs in CI
// and validates every op + tensor layout in the graph.  It exercises the ggml
// CPU backend always, and additionally any registered GPU backend (Vulkan on
// Windows/Linux, Metal, CUDA, OpenCL) — so the same binary validates the GPU
// port on capable machines (e.g. an RTX 5090 Vulkan device) and degrades to a
// CPU-only parity check elsewhere.
//
// When a fixtures directory (argv[1], from scripts/dump-lavasr-enhancer-fixtures.py)
// is present it additionally compares the ggml forward against the onnxruntime
// golden real/imag with the real model weights.

#include "backend_selection.h"
#include "lavasr/enhancer.h"      // scalar enhance() + enhance_with()
#include "lavasr/enhancer_core.h"
#include "lavasr/enhancer_ggml.h"
#include "npy.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using tts_cpp::lavasr::EnhancerGgml;
using tts_cpp::lavasr::EnhancerWeights;
using tts_cpp::lavasr::EnhTensor;

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

struct Diff {
    float  max_abs = 0.0f;
    float  rel     = 0.0f;
    double cos_sim = 0.0;
    size_t n       = 0;
    bool   size_ok = true;
    bool   finite  = true;
};

static Diff compare(const std::vector<float> & a, const std::vector<float> & b) {
    Diff d;
    if (a.size() != b.size()) {
        d.size_ok = false;
        return d;
    }
    d.n = a.size();
    float  maxd = 0.0f, maxg = 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            d.finite = false;
        }
        maxd = std::max(maxd, std::fabs(a[i] - b[i]));
        maxg = std::max(maxg, std::fabs(b[i]));
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    d.max_abs = maxd;
    d.rel     = maxd / (maxg + 1e-9f);
    d.cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    return d;
}

// Run the graph on `backend`, compare real/imag to the scalar oracle (and, when
// provided, the onnxruntime golden).  Returns the number of failures.
static int run_backend(const char * label, ggml_backend_t backend,
                       const EnhancerWeights & w, const std::vector<float> & mel, int T,
                       const std::vector<float> & re_ref, const std::vector<float> & im_ref,
                       const char * ref_name, float tol,
                       const std::vector<float> * re_gold = nullptr,
                       const std::vector<float> * im_gold = nullptr,
                       float tol_gold = 0.0f) {
    EnhancerGgml * g = tts_cpp::lavasr::enhancer_ggml_create(w, backend);
    if (!g) {
        std::fprintf(stderr, "FAIL[%s]: enhancer_ggml_create failed\n", label);
        return 1;
    }
    std::vector<float> re, im;
    const bool         ok = tts_cpp::lavasr::enhancer_ggml_spec_forward(g, mel, T, re, im);
    tts_cpp::lavasr::enhancer_ggml_free(g);
    if (!ok) {
        std::fprintf(stderr, "FAIL[%s]: enhancer_ggml_spec_forward failed\n", label);
        return 1;
    }

    int  failures = 0;
    auto check     = [&](const char * what, const std::vector<float> & got,
                     const std::vector<float> & ref, float t) {
        Diff d = compare(got, ref);
        if (!d.size_ok) {
            std::fprintf(stderr, "FAIL[%s]: %s size mismatch (%zu vs %zu)\n", label, what,
                         got.size(), ref.size());
            ++failures;
            return;
        }
        std::printf("  [%s] %-18s max_abs=%.3e rel=%.3e cos_sim=%.7f (n=%zu)\n", label, what,
                    d.max_abs, d.rel, d.cos_sim, d.n);
        if (!d.finite) {
            std::fprintf(stderr, "FAIL[%s]: %s produced non-finite values\n", label, what);
            ++failures;
        }
        // A layout / indexing bug tanks cos_sim well below 1 even if a few
        // clipped outliers keep max_abs deceptively small — gate on both.
        if (!(d.max_abs < t) || d.cos_sim < 0.9999) {
            std::fprintf(stderr, "FAIL[%s]: %s exceeds tol (max_abs=%.3e tol=%.0e cos_sim=%.7f)\n",
                         label, what, d.max_abs, t, d.cos_sim);
            ++failures;
        }
    };

    const std::string rn = ref_name;
    check(("real vs " + rn).c_str(), re, re_ref, tol);
    check(("imag vs " + rn).c_str(), im, im_ref, tol);
    if (re_gold && im_gold) {
        check("real vs golden", re, *re_gold, tol_gold);
        check("imag vs golden", im, *im_gold, tol_gold);
    }
    return failures;
}

// Runs scalar oracle + ggml CPU (+ GPU if present) and compares.  `ref_name`
// labels the scalar reference in the output.  Returns failure count.
static int parity_over_backends(const EnhancerWeights & w, const std::vector<float> & mel,
                                int T, const std::vector<float> * re_gold = nullptr,
                                const std::vector<float> * im_gold = nullptr) {
    std::vector<float> re_scalar, im_scalar;
    tts_cpp::lavasr::enhancer_spec_forward(w, mel, T, re_scalar, im_scalar);

    int failures = 0;

    ggml_backend_t cpu = tts_cpp::detail::init_cpu_backend();
    if (!cpu) {
        std::fprintf(stderr, "FAIL: init_cpu_backend returned null\n");
        return 1;
    }
    // ggml-cpu vs the scalar core: identical arithmetic, different summation
    // order + ggml's own erf/exp/sin/cos — comfortably inside the 3e-3 band the
    // scalar core is validated against the onnx golden at.
    failures += run_backend("cpu", cpu, w, mel, T, re_scalar, im_scalar, "scalar",
                            /*tol=*/3e-3f, re_gold, im_gold, /*tol_gold=*/3e-3f);
    ggml_backend_free(cpu);

    ggml_backend_t gpu = tts_cpp::detail::init_gpu_backend(
        /*n_gpu_layers=*/99, /*verbose=*/true, "test-enhancer", /*vulkan_device=*/0);
    if (gpu) {
        // f32 GPU rounding (FMA, different reductions) drifts a little more from
        // the scalar core; still far tighter than any layout bug would allow.
        failures += run_backend("gpu", gpu, w, mel, T, re_scalar, im_scalar, "scalar",
                                /*tol=*/6e-3f, re_gold, im_gold, /*tol_gold=*/7e-3f);
        ggml_backend_free(gpu);
    } else {
        std::printf("  [gpu] no GPU backend registered — CPU-only parity check\n");
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Self-contained random-weight enhancer
// ---------------------------------------------------------------------------

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

// Fill w.t with deterministic pseudo-random weights sized to w's dims.  Small
// conv/linear weights keep activations well-conditioned (log-mag stays modest so
// exp doesn't saturate to the clip in every bin).
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

static int selftest() {
    // Small but structurally complete enhancer (grouped embed conv, depthwise
    // conv, LN, pointwise convs, gelu, gamma, residual, spec head).
    EnhancerWeights w;
    w.dim       = 32;
    w.ffn_dim   = 64;
    w.n_blocks  = 3;
    w.n_mels    = 16;
    w.kernel    = 7;
    w.spec_bins = 24;
    w.clip_max  = 1000.0f;
    w.ln_eps    = 1e-6f;

    const int C = w.dim, F = w.ffn_dim, M = w.n_mels, B = w.spec_bins;
    const int T = 40;

    std::mt19937 rng(1234567u);
    fill_random_weights(w, rng);

    std::vector<float>              mel(static_cast<size_t>(M) * T);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : mel) {
        v = nd(rng);
    }

    std::printf("LavaSR enhancer GGML self-test (random weights, T=%d, C=%d, F=%d, blocks=%d, M=%d, B=%d):\n",
                T, C, F, w.n_blocks, M, B);
    return parity_over_backends(w, mel, T);
}

// ---------------------------------------------------------------------------
// Full enhance() pipeline parity (scalar core vs ggml graph)
// ---------------------------------------------------------------------------
// Exercises the end-to-end enhance() glue — resample -> log-mel -> neural core
// -> ISTFT -> FastLR crossover — comparing the scalar path against the ggml
// graph path on the default CPU backend (and any GPU backend).  This is the code
// path Enhancer::load() now takes for use_gpu=false, so it guards the ggml-CPU
// default in addition to the neural-core parity above.  Self-contained.

static int check_pipeline(const char * label, const std::vector<float> & got,
                          const std::vector<float> & ref) {
    Diff d = compare(got, ref);
    if (!d.size_ok) {
        std::fprintf(stderr, "FAIL[%s]: enhance() size mismatch (%zu vs %zu)\n", label,
                     got.size(), ref.size());
        return 1;
    }
    std::printf("  [%s] enhance() vs scalar  max_abs=%.3e rel=%.3e cos_sim=%.7f (n=%zu)\n",
                label, d.max_abs, d.rel, d.cos_sim, d.n);
    int failures = 0;
    if (!d.finite) {
        std::fprintf(stderr, "FAIL[%s]: enhance() produced non-finite samples\n", label);
        ++failures;
    }
    // The DSP stages are shared, so only the core's f32 rounding differs; a
    // layout/wiring regression in the graph path tanks cos_sim well below this.
    if (d.cos_sim < 0.999) {
        std::fprintf(stderr, "FAIL[%s]: enhance() cos_sim %.7f below 0.999\n", label, d.cos_sim);
        ++failures;
    }
    return failures;
}

static int run_pipeline_backend(const char * label, ggml_backend_t backend,
                                const EnhancerWeights & w, const std::vector<float> & pcm,
                                int sr_in, const std::vector<float> & ref) {
    EnhancerGgml * g = tts_cpp::lavasr::enhancer_ggml_create(w, backend);
    if (!g) {
        std::fprintf(stderr, "FAIL[%s]: enhancer_ggml_create failed\n", label);
        return 1;
    }
    std::vector<float> out = tts_cpp::lavasr::enhance(w, g, pcm, sr_in);
    tts_cpp::lavasr::enhancer_ggml_free(g);
    return check_pipeline(label, out, ref);
}

static int selftest_pipeline() {
    // Small network but production-consistent DSP dims (spec_bins = n_fft/2+1,
    // default n_fft=2048 / hop=512 / 80 mels / 48k work rate) so the resample ->
    // mel -> ISTFT -> crossover pipeline runs exactly as in production.
    EnhancerWeights w;
    w.dim      = 64;
    w.ffn_dim  = 128;
    w.n_blocks = 2;

    std::mt19937 rng(20260706u);
    fill_random_weights(w, rng);

    // ~0.5 s of 24 kHz mono input (low tone + a high tone for the extended band).
    const int          sr_in = 24000;
    const int          n     = sr_in / 2;
    std::vector<float> pcm(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        const float ph = 2.0f * 3.14159265358979f * i / sr_in;
        pcm[i]         = 0.1f * std::sin(220.0f * ph) + 0.05f * std::sin(3000.0f * ph);
    }

    const std::vector<float> ref = tts_cpp::lavasr::enhance(w, pcm, sr_in); // scalar oracle

    std::printf("LavaSR enhancer full-pipeline parity (C=%d, blocks=%d, n_fft=%d, input=%.2fs @ %dHz, out=%zu):\n",
                w.dim, w.n_blocks, w.n_fft, static_cast<double>(n) / sr_in, sr_in, ref.size());

    int failures = 0;
    ggml_backend_t cpu = tts_cpp::detail::init_cpu_backend();
    if (!cpu) {
        std::fprintf(stderr, "FAIL: init_cpu_backend returned null\n");
        return 1;
    }
    failures += run_pipeline_backend("cpu", cpu, w, pcm, sr_in, ref);
    ggml_backend_free(cpu);

    ggml_backend_t gpu = tts_cpp::detail::init_gpu_backend(
        /*n_gpu_layers=*/99, /*verbose=*/false, "test-enh-pipe", /*vulkan_device=*/0);
    if (gpu) {
        failures += run_pipeline_backend("gpu", gpu, w, pcm, sr_in, ref);
        ggml_backend_free(gpu);
    } else {
        std::printf("  [gpu] no GPU backend registered — CPU-only pipeline check\n");
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Optional onnxruntime-golden fixture path
// ---------------------------------------------------------------------------

static std::vector<float> load_f32(const std::string & path, std::vector<int> * shape = nullptr) {
    npy_array a = npy_load(path);
    if (a.dtype != "<f4") {
        throw std::runtime_error("expected <f4 in " + path + " got " + a.dtype);
    }
    const size_t       n = a.n_elements();
    std::vector<float> v(n);
    std::memcpy(v.data(), a.data.data(), n * sizeof(float));
    if (shape) {
        shape->clear();
        for (auto d : a.shape) {
            shape->push_back(static_cast<int>(d));
        }
    }
    return v;
}

static bool file_exists(const std::string & path) {
    if (FILE * f = std::fopen(path.c_str(), "rb")) {
        std::fclose(f);
        return true;
    }
    return false;
}

static int golden_test(const std::string & dir) {
    auto load_weight = [&](EnhancerWeights & w, const std::string & name) {
        EnhTensor        t;
        std::vector<int> shape;
        t.data    = load_f32(dir + "/" + name + ".npy", &shape);
        t.shape   = shape;
        w.t[name] = std::move(t);
    };

    EnhancerWeights w;
    load_weight(w, "enhancer.embed.weight");
    load_weight(w, "enhancer.embed.bias");
    load_weight(w, "enhancer.norm.weight");
    load_weight(w, "enhancer.norm.bias");
    for (int i = 0; i < w.n_blocks; i++) {
        const std::string p = "enhancer.block." + std::to_string(i) + ".";
        for (const char * s : {"dwconv.weight", "dwconv.bias", "norm.weight", "norm.bias",
                               "pwconv1.weight", "pwconv1.bias", "pwconv2.weight",
                               "pwconv2.bias", "gamma"}) {
            load_weight(w, p + s);
        }
    }
    load_weight(w, "enhancer.final_norm.weight");
    load_weight(w, "enhancer.final_norm.bias");
    load_weight(w, "spec_head.out.weight");
    load_weight(w, "spec_head.out.bias");

    std::vector<int>   mel_shape;
    std::vector<float> mel = load_f32(dir + "/mel.npy", &mel_shape);
    if (mel_shape.size() != 2 || mel_shape[0] != w.n_mels) {
        std::fprintf(stderr, "FAIL: mel shape unexpected\n");
        return 1;
    }
    const int          T       = mel_shape[1];
    std::vector<float> re_gold = load_f32(dir + "/real.npy");
    std::vector<float> im_gold = load_f32(dir + "/imag.npy");

    std::printf("LavaSR enhancer GGML golden parity (T=%d, C=%d, blocks=%d):\n", T, w.dim,
                w.n_blocks);
    return parity_over_backends(w, mel, T, &re_gold, &im_gold);
}

int main(int argc, char ** argv) {
    int failures = selftest();
    failures += selftest_pipeline();

    // Bonus: when the real fixtures are present, also check the ggml forward
    // against the onnxruntime golden with the production weights.
    if (argc >= 2) {
        const std::string dir = argv[1];
        if (file_exists(dir + "/real.npy") && file_exists(dir + "/mel.npy")) {
            try {
                failures += golden_test(dir);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "FAIL: golden fixture test: %s\n", e.what());
                ++failures;
            }
        } else {
            std::printf("(fixtures dir '%s' has no real.npy/mel.npy — skipping golden path)\n",
                        dir.c_str());
        }
    }

    if (failures == 0) {
        std::printf("OK: enhancer GGML forward matches the scalar core across backends\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d check(s)\n", failures);
    return 1;
}
