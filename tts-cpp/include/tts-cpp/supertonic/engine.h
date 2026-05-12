#pragma once

// Persistent Supertonic engine.
//
// Loads the Supertonic GGUF once, validates the requested voice / language /
// step count, and keeps the model resident so subsequent calls to
// `synthesize()` only pay the per-call preprocess + duration + text-encoder +
// vector-estimator + vocoder cost - not the GGUF tensor load.
//
// Usage:
//
//     using tts_cpp::supertonic::Engine;
//     using tts_cpp::supertonic::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_gguf_path = "models/supertonic.gguf";
//     opts.n_gpu_layers    = 0;                      // CPU only today
//
//     Engine engine(opts);
//     for (const auto & line : lines) {
//         auto result = engine.synthesize(line);
//         write_wav(result.pcm, result.sample_rate);
//     }
//
// Threading model:
//   - synthesize() on the same Engine instance is NOT safe to call
//     concurrently - the per-stage thread_local caches and the seeded
//     RNG are per-instance shared state.
//   - synthesize() on different Engine instances from different
//     threads is safe.  The supertonic generation_id (set per Engine
//     ctor) keys the stage-internal caches so two Engines don't collide.
//   - cancel() is safe from any thread.
//
// Implemented in src/supertonic_engine.cpp on top of the library-internal
// helpers in src/supertonic_internal.h.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::supertonic {

struct EngineOptions {
    // Required.
    std::string model_gguf_path;

    // Empty / zero values use the defaults stored in the GGUF metadata.
    std::string voice;
    std::string language = "en";
    int   steps    = 0;
    float speed    = 0.0f;
    int   seed     = 42;
    int   n_threads     = 0;
    int   n_gpu_layers  = 0;

    // F16 K/V flash-attention in the vector estimator.  When -1, the
    // engine auto-enables this on GPU backends (non-CPU) and disables
    // it on CPU; pass 1 / 0 to force the setting regardless of the
    // resolved backend.  Triggers the OpenCL `flash_attn_f32_f16`
    // path on Adreno; mirrors chatterbox's `--cfm-f16-kv-attn`.  No
    // effect on CPU (the cblas attention path is already efficient).
    // On Vulkan dispatches `kernel_flash_attn_f32_f16_*` (head_dim=64
    // satisfies the `HSK % 8 == 0` supports_op gate; see
    // `ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT`).
    int f16_attn = -1;

    // QVAC-18605 — Vulkan adapter index.  Passed verbatim to
    // `ggml_backend_vk_init(idx)` when the build is compiled with
    // `GGML_VULKAN=ON` and `n_gpu_layers > 0`.  Range-checked
    // against `ggml_backend_vk_get_device_count()` at load; an
    // out-of-range value throws (no silent CPU fallback — that
    // would mask CLI typos / wrong-machine config).  Default 0
    // (the historical hard-coded value).  Negative values are
    // reserved for a future "auto-pick best device" policy.
    int vulkan_device = 0;

    // F16 storage type for the audit-identified hot matmul /
    // pointwise-conv weights (vector-estimator attention W_*,
    // pwconv1/pwconv2 across every convnext block, vocoder
    // head linear, text-encoder linears, …).  Same -1/0/1 tri-state
    // as `f16_attn`: -1 auto (on for GPU, off for CPU); 0 or 1 force.
    // Halves the GPU read bandwidth into those ops with a small
    // (≤ 2e-3 abs / 5e-3 cosine) numerical drift on the end-to-end
    // synth.  Mirrors chatterbox's CHATTERBOX_F16_CFM gate.
    int f16_weights = -1;

    // Optional path to a .npy file containing the initial noise tensor of
    // shape [1, latent_channels, latent_len] (float32).  When provided,
    // latent_len is taken from the npy file (overriding the duration-
    // predicted length) and the seeded RNG is bypassed.  Useful for
    // byte-exact reproduction of an ONNX/PyTorch reference run.
    std::string noise_npy_path;

    // QVAC-18605 follow-up — first-synth-latency pre-warming.
    //
    // When non-empty, the Engine ctor invokes `warm_up(prewarm_text)`
    // immediately after the GGUF load + voice validation, running one
    // throwaway synth on the supplied text.  On Vulkan / OpenCL this
    // forces the GPU shader pipelines for every Supertonic stage to
    // compile up-front (the in-tree thread_local graph caches handle
    // every subsequent call but can't avoid the first pipeline-compile
    // cost — measured ~hundreds of ms on first synth on Adreno + RADV
    // in chatterbox PROGRESS.md), so the operator-visible first synth
    // call sees ~steady-state latency.  No effect on CPU (no shader
    // compilation cost; warm_up returns immediately on
    // `model.backend_is_cpu`).
    //
    // Pre-warm text should be similar in length to representative
    // production input — the per-stage graph caches are keyed on
    // (text_len, latent_len) tuples, so a too-short pre-warm leaves
    // a graph-rebuild on the first real call (still saves the
    // shader-compile cost; only the cgraph allocation is repeated).
    // Default empty (no pre-warming).
    std::string prewarm_text;
};

struct SynthesisResult {
    std::vector<float> pcm;
    int   sample_rate = 44100;
    float duration_s  = 0.0f;
};

// Persistent engine.  Loads the GGUF once at construction; subsequent
// synthesize() calls reuse the resident model.
class TTS_CPP_API Engine {
public:
    // Loads the Supertonic GGUF, initialises the backend, validates
    // opts.voice / opts.language up front.  Throws std::runtime_error
    // on any hard failure (GGUF not found, GGUF malformed, unsupported
    // voice).
    explicit Engine(const EngineOptions & opts);

    // Frees the backend + all ggml contexts.
    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;

    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize `text` into PCM (44.1 kHz mono float32 by default;
    // see SynthesisResult::sample_rate).  Throws std::runtime_error
    // on failure.  Empty `text` is rejected.
    //
    // Not safe to call concurrently on the same Engine instance.
    SynthesisResult synthesize(const std::string & text);

    // Best-effort cancel of an in-flight synthesize() call on another
    // thread.  Setting the flag is all this does; actual termination
    // happens at the next cancellation check inside the vector-
    // estimator loop (one step is the worst-case cancel latency).
    void cancel();

    // QVAC-18605 follow-up — first-synth-latency pre-warming.
    //
    // Runs one throwaway synth on `text` to force every per-stage
    // GPU graph cache to populate and every Vulkan / OpenCL shader
    // pipeline to compile up-front.  The PCM result is discarded.
    // Subsequent `synthesize()` calls hit the warmed caches +
    // pre-compiled pipelines, so the operator-visible first synth
    // sees steady-state latency.
    //
    // No-op on CPU backends (no pipeline cache to warm).  Auto-
    // invoked by the ctor when `EngineOptions::prewarm_text` is
    // non-empty; callers can also invoke explicitly mid-life when
    // they need to warm a different shape (e.g. switching from a
    // short-prompt to a long-prompt workload).
    //
    // Throws on the same conditions as `synthesize()` — if the
    // throwaway synth fails for any reason, the failure surfaces
    // here rather than being swallowed.
    void warm_up(const std::string & text);

    // Return the options the engine was constructed with (convenience
    // for callers that want to introspect the resolved n_gpu_layers /
    // n_threads after defaults are applied).
    const EngineOptions & options() const;

    // Return the registered name of the backend the engine actually
    // resolved to during construction (e.g. "CPU", "Metal").  Returns
    // "(unknown)" when the backend is unset.
    std::string backend_name() const;

    // Resolved compute device.  CPU when the build has no GPU backend
    // compiled in, when no GPU was requested (n_gpu_layers <= 0), or
    // when the requested GPU backend refused to initialise.  GPU
    // otherwise.  Stable for the lifetime of the Engine.
    BackendDevice backend_device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience one-shot wrapper around Engine.  Equivalent to:
//   Engine e(opts); return e.synthesize(text);
// Use the Engine class directly for any host that synthesizes more
// than once - this wrapper pays the full GGUF load + free per call.
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::supertonic
