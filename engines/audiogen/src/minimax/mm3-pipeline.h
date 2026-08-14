#pragma once

#include "mm3-ar-loop.h"
#include "mm3-cond-graph.h"
#include "mm3-dit-graph.h"
#include "mm3-model.h"
#include "mm3-tokenizer.h"
#include "mm3-vocoder-graph.h"
#include "logic.h"
#include "progress.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#define MM3_OVERLAP_LATENTS    172
#define MM3_CARRY_SPAN_LATENTS 344
#define MM3_CROP_LEFT_LATENTS  86
#define MM3_CROP_RIGHT_LATENTS 258

static bool mm3_flow_sample_chunk(const MM3Model & m, const float * noise, const float * cond, int64_t L, int steps,
                                  float cfg_scale, int64_t overlap, const float * prev_latent, int64_t prev_stride,
                                  std::vector<float> & out_latents, MM3FlowStats * stats,
                                  const std::function<void(int, int)> & on_step,
                                  const std::function<bool()> & should_cancel, std::string * err) {
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_DIT_MAX_FRAMES);
        }
        return false;
    }
    if (steps <= 0 || steps > 1000) {
        if (err) {
            *err = "steps must be in 1..1000";
        }
        return false;
    }
    if (overlap < 0 || overlap > L) {
        if (err) {
            *err = "overlap must be in 0..L";
        }
        return false;
    }
    if (overlap > 0 && (!prev_latent || prev_stride < overlap)) {
        if (err) {
            *err = "a positive overlap needs a previous-window carry at least that long";
        }
        return false;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, err)) {
        return false;
    }

    const int64_t C = (int64_t) m.synth_cfg.dit.in_channels;
    const int64_t N = C * L;
    out_latents.assign((size_t) N, 0.0f);
    memcpy(out_latents.data(), noise, (size_t) N * sizeof(float));

    std::vector<float> sigmas, timesteps;
    mm3_flow_sigmas(steps, &sigmas, &timesteps);

    std::vector<float> pred_c((size_t) N);
    std::vector<float> pred_u((size_t) N);

    const auto t_all  = std::chrono::steady_clock::now();
    double     fwd_ms = 0.0, first_ms = 0.0, last_ms = 0.0;

    for (int i = 0; i < steps; i++) {
        if (tts_cpp::minimax::detail::cancellation_requested(should_cancel)) {
            if (err) {
                *err = MM3_ERR_CANCELLED;
            }
            return false;
        }
        const float t = timesteps[(size_t) i];

        if (overlap > 0) {
            const float a = 1.0f - (1.0f - 1e-6f) * t;
            for (int64_t c = 0; c < C; c++) {
                const float * np = noise + c * L;
                const float * pl = prev_latent + c * prev_stride;
                float *       x  = out_latents.data() + c * L;
                for (int64_t j = 0; j < overlap; j++) {
                    x[j] = a * np[j] + t * pl[j];
                }
            }
        }

        const auto t0 = std::chrono::steady_clock::now();

        if (!mm3_dit_run(m, &g_mm3_dit, out_latents.data(), i == 0 ? cond : nullptr, 1.0f, t, L, pred_c.data(), err)) {
            return false;
        }

        if (tts_cpp::minimax::detail::cancellation_requested(should_cancel)) {
            if (err) {
                *err = MM3_ERR_CANCELLED;
            }
            return false;
        }

        if (!mm3_dit_run(m, &g_mm3_dit, out_latents.data(), nullptr, 0.0f, t, L, pred_u.data(), err)) {
            return false;
        }

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fwd_ms += ms;
        if (i == 0) {
            first_ms = ms;
        }
        last_ms = ms;

        const float dsigma = sigmas[(size_t) i + 1] - sigmas[(size_t) i];
        for (int64_t j = 0; j < N; j++) {
            const float u = pred_u[(size_t) j];
            const float v = u + cfg_scale * (pred_c[(size_t) j] - u);
            out_latents[(size_t) j] += dsigma * v;
        }

        if (on_step) {
            on_step(i + 1, steps);
        }

        if (should_cancel && should_cancel()) {
            if (err) {
                *err = MM3_ERR_CANCELLED;
            }
            return false;
        }
    }

    if (overlap > 0) {
        for (int64_t c = 0; c < C; c++) {
            memcpy(out_latents.data() + c * L, prev_latent + c * prev_stride, (size_t) overlap * sizeof(float));
        }
    }

    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all).count();
    if (stats) {
        stats->steps         = steps;
        stats->forwards      = steps * 2;
        stats->total_ms      = total_ms;
        stats->forward_ms    = fwd_ms;
        stats->first_ms      = first_ms;
        stats->last_ms       = last_ms;
        stats->compute_bytes = g_mm3_dit.compute_bytes;
    }
    fprintf(stderr,
            "[MM3-Flow] Window: L=%lld, ov=%lld, %d steps, cfg %.2f -> %.0f ms (%.0f ms/step, %.0f ms/forward)\n",
            (long long) L, (long long) overlap, steps, (double) cfg_scale, total_ms, total_ms / (double) steps,
            fwd_ms / (double) (steps * 2));
    return true;
}

struct MM3GenRequest {

    std::string          prompt;
    std::vector<int32_t> ids_cond;
    std::vector<int32_t> ids_uncond;

    int64_t  max_frames = 300;
    uint64_t seed       = 42;
    int      steps      = 30;
    float    cfg_flow   = 1.7f;

    std::vector<int32_t> forced_semantic;
    std::vector<int32_t> forced_acoustic;

    std::vector<std::vector<float>> forced_noise;

    bool keep_window_latents = false;

    std::function<bool()> should_cancel;
};

struct MM3GenResult {
    std::vector<float> audio;
    int64_t            n_samples = 0;
    int                sample_rate = 0;

    int64_t              frames    = 0;
    int64_t              n_windows = 0;
    std::vector<int64_t> chunk_starts;
    std::vector<int64_t> chunk_frames;
    std::vector<int64_t> window_L;
    std::vector<int64_t> window_overlap;
    std::vector<int64_t> forced_noise_used;

    std::vector<std::vector<float>> window_latents;

    MM3ArResult ar;

    double  ar_ms    = 0.0;
    double  cond_ms  = 0.0;
    double  flow_ms  = 0.0;
    double  voc_ms   = 0.0;
    double  total_ms = 0.0;
    int64_t flow_forwards = 0;

    size_t dit_compute_bytes = 0;

    bool  has_nan = false;
    float peak    = 0.0f;
    double rms    = 0.0;
};

struct MM3PipelineDimensions {
    int64_t acoustic_codebooks = 0;
    int64_t hidden = 0;
    int64_t layers = 0;
    int64_t channels = 0;
    int64_t upsample = 0;
    int64_t window_frames = 0;
    int64_t hop_frames = 0;
};

struct MM3WindowCarry {
    std::vector<float> latent;
    std::vector<float> condition;
    int64_t length = 0;
};

constexpr int kMm3StereoChannels = 2;
constexpr size_t kMm3CancellationPollSamples = 4096;

static bool mm3_fail(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

static MM3PipelineDimensions mm3_pipeline_dimensions(const MM3Model & model) {
    MM3PipelineDimensions dimensions;
    dimensions.acoustic_codebooks = static_cast<int64_t>(model.lm_cfg.num_codebooks) - 1;
    dimensions.hidden = static_cast<int64_t>(model.lm_cfg.embedding_length);
    dimensions.layers = static_cast<int64_t>(model.lm_cfg.num_codebooks);
    dimensions.channels = static_cast<int64_t>(model.synth_cfg.dit.in_channels);
    dimensions.upsample = static_cast<int64_t>(model.synth_cfg.voc.total_upsample);
    dimensions.window_frames = static_cast<int64_t>(model.synth_cfg.dit.window_frames);
    dimensions.hop_frames = static_cast<int64_t>(model.synth_cfg.dit.hop_frames);
    return dimensions;
}

static bool mm3_prepare_token_ids(const MM3Model & model, const MM3GenRequest & request,
                                  MM3Tokenizer * tokenizer, std::vector<int32_t> & conditional,
                                  std::vector<int32_t> & unconditional, std::string * error) {
    conditional = request.ids_cond;
    unconditional = request.ids_uncond;
    if (conditional.empty()) {
        if (request.prompt.empty()) {
            return mm3_fail(error, "need either a prompt or ids_cond");
        }
        if (!tokenizer || !mm3_tokenizer_load(model, tokenizer, error)) {
            return false;
        }
        mm3_tokenizer_encode(*tokenizer, request.prompt, &conditional);
    }
    if (conditional.size() < 3) {
        return mm3_fail(error, "the prompt must tokenise to at least 3 tokens");
    }
    if (conditional.size() > model.lm_cfg.max_prompt_tokens) {
        return mm3_fail(error, "the prompt exceeds the model token limit");
    }
    if (unconditional.empty()) {
        mm3_tokenizer_uncond(model.lm_cfg, conditional, &unconditional);
    }
    if (unconditional.size() != conditional.size()) {
        return mm3_fail(error, "ids_uncond must be the same length as ids_cond");
    }
    return true;
}

static bool mm3_prepare_ar_options(const MM3GenRequest & request, int64_t acoustic_codebooks,
                                   const MM3ProgressCb & progress, MM3ArOptions & options,
                                   std::string * error) {
    options.max_frames = request.max_frames;
    options.seed = request.seed;
    options.collect_hiddens = true;
    options.should_cancel = request.should_cancel;
    if (!request.forced_semantic.empty()) {
        const int64_t expected =
            static_cast<int64_t>(request.forced_semantic.size()) * acoustic_codebooks;
        if (static_cast<int64_t>(request.forced_acoustic.size()) != expected) {
            return mm3_fail(error, "forced_acoustic must contain " +
                                       std::to_string(acoustic_codebooks) +
                                       " entries per semantic token");
        }
        options.forced_semantic = request.forced_semantic.data();
        options.forced_acoustic = request.forced_acoustic.data();
        options.forced_len = static_cast<int64_t>(request.forced_semantic.size());
    }
    if (progress) {
        options.on_frame = [progress](int64_t frame, int64_t total) {
            progress(MM3GenProgress{"ar", -1, 0, frame, total});
        };
    }
    return true;
}

static bool mm3_run_ar_stage(const MM3Model & model, const MM3GenRequest & request,
                             const MM3PipelineDimensions & dimensions,
                             const std::vector<int32_t> & conditional,
                             const std::vector<int32_t> & unconditional,
                             const MM3ProgressCb & progress, MM3GenResult * result,
                             std::string * error) {
    MM3ArOptions options;
    if (!mm3_prepare_ar_options(request, dimensions.acoustic_codebooks, progress, options, error)) {
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    if (!mm3_ar_plan(model, conditional.data(), unconditional.data(),
                     static_cast<int64_t>(conditional.size()), options, &result->ar, error)) {
        return false;
    }
    result->ar_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    result->frames = result->ar.n_frames;
    if (result->frames <= 0) {
        return mm3_fail(error, "the AR stage emitted zero frames");
    }
    const int64_t expected = result->frames * dimensions.layers * dimensions.hidden;
    if (static_cast<int64_t>(result->ar.frame_hiddens.size()) != expected) {
        return mm3_fail(error, "the AR stage returned a frame-hidden block of the wrong size");
    }
    return true;
}

static bool mm3_encode_window_condition(const MM3Model & model, const MM3GenRequest & request,
                                        const MM3PipelineDimensions & dimensions,
                                        const MM3ProgressCb & progress, MM3GenResult * result,
                                        int64_t window, int64_t window_count, int64_t start,
                                        int64_t frames, std::vector<float> & condition,
                                        int64_t & latent_length, std::string * error) {
    if (!mm3_emit_progress(progress, {"cond", window, window_count, 0, 1},
                           request.should_cancel, error)) {
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    const size_t hidden_offset =
        static_cast<size_t>(start * dimensions.layers * dimensions.hidden);
    if (!mm3_cond_encode(model, result->ar.frame_hiddens.data() + hidden_offset, frames,
                         condition, &latent_length, error)) {
        return false;
    }
    result->cond_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    result->window_L.push_back(latent_length);
    return true;
}

static int64_t mm3_apply_window_carry(const MM3Model & model, const MM3WindowCarry & carry,
                                      int64_t latent_length, std::vector<float> & condition) {
    const int64_t overlap =
        carry.length > 0 ? std::min<int64_t>(carry.length, latent_length) : 0;
    if (overlap > 0) {
        const size_t bytes =
            static_cast<size_t>(overlap * static_cast<int64_t>(model.synth_cfg.cond.out_dim)) *
            sizeof(float);
        memcpy(condition.data(), carry.condition.data(), bytes);
    }
    return overlap;
}

static bool mm3_prepare_window_noise(const MM3GenRequest & request, int64_t window,
                                     int64_t count, std::vector<float> & noise) {
    if (static_cast<int64_t>(request.forced_noise.size()) > window &&
        static_cast<int64_t>(request.forced_noise[static_cast<size_t>(window)].size()) == count) {
        noise = request.forced_noise[static_cast<size_t>(window)];
        return true;
    }
    tts_cpp::minimax::detail::fill_noise(request.seed, window, noise, count);
    return false;
}

static bool mm3_sample_window(const MM3Model & model, const MM3GenRequest & request,
                              const MM3ProgressCb & progress, const std::vector<float> & noise,
                              const std::vector<float> & condition, const MM3WindowCarry & carry,
                              int64_t window, int64_t window_count, int64_t latent_length,
                              int64_t overlap, std::vector<float> & latents,
                              MM3GenResult * result, std::string * error) {
    if (!mm3_emit_progress(progress, {"flow", window, window_count, 0, request.steps},
                           request.should_cancel, error)) {
        return false;
    }
    const auto on_step = [progress, window, window_count](int step, int total) {
        if (progress) {
            progress(MM3GenProgress{"flow", window, window_count, step, total});
        }
    };
    MM3FlowStats stats;
    if (!mm3_flow_sample_chunk(model, noise.data(), condition.data(), latent_length,
                               request.steps, request.cfg_flow, overlap,
                               overlap > 0 ? carry.latent.data() : nullptr, carry.length,
                               latents, &stats, on_step, request.should_cancel, error)) {
        return false;
    }
    result->flow_ms += stats.total_ms;
    result->flow_forwards += stats.forwards;
    result->dit_compute_bytes = stats.compute_bytes;
    return true;
}

static void mm3_copy_carry_latents(const std::vector<float> & latents, int64_t channels,
                                   int64_t latent_length, int64_t carry_start,
                                   MM3WindowCarry & carry) {
    for (int64_t channel = 0; channel < channels; ++channel) {
        memcpy(carry.latent.data() + channel * carry.length,
               latents.data() + channel * latent_length + carry_start,
               static_cast<size_t>(carry.length) * sizeof(float));
    }
}

static void mm3_update_window_carry(const MM3Model & model,
                                    const MM3PipelineDimensions & dimensions,
                                    const std::vector<float> & latents,
                                    const std::vector<float> & condition,
                                    int64_t latent_length, MM3WindowCarry & carry) {
    const int64_t start =
        std::max<int64_t>(latent_length - MM3_CARRY_SPAN_LATENTS, 0);
    const int64_t end =
        std::max<int64_t>(latent_length - MM3_OVERLAP_LATENTS, start);
    carry.length = end - start;
    if (carry.length <= 0) {
        carry.latent.clear();
        carry.condition.clear();
        return;
    }
    carry.latent.assign(static_cast<size_t>(dimensions.channels * carry.length), 0.0f);
    mm3_copy_carry_latents(latents, dimensions.channels, latent_length, start, carry);
    const int64_t condition_dimension = static_cast<int64_t>(model.synth_cfg.cond.out_dim);
    carry.condition.assign(static_cast<size_t>(carry.length * condition_dimension), 0.0f);
    memcpy(carry.condition.data(), condition.data() + start * condition_dimension,
           static_cast<size_t>(carry.length * condition_dimension) * sizeof(float));
}

static bool mm3_generate_window(const MM3Model & model, const MM3GenRequest & request,
                                const MM3PipelineDimensions & dimensions,
                                const MM3ProgressCb & progress, MM3GenResult * result,
                                const std::vector<int64_t> & starts, int64_t window,
                                MM3WindowCarry & carry, std::vector<float> & latents,
                                std::string * error) {
    const int64_t window_count = static_cast<int64_t>(starts.size());
    const int64_t start = starts[static_cast<size_t>(window)];
    const int64_t end = std::min<int64_t>(start + dimensions.window_frames, result->frames);
    const int64_t frames = end - start;
    result->chunk_frames.push_back(frames);
    std::vector<float> condition;
    int64_t latent_length = 0;
    if (!mm3_encode_window_condition(model, request, dimensions, progress, result, window,
                                     window_count, start, frames, condition, latent_length, error)) {
        return false;
    }
    const int64_t overlap =
        mm3_apply_window_carry(model, carry, latent_length, condition);
    result->window_overlap.push_back(overlap);
    std::vector<float> noise;
    const bool forced = mm3_prepare_window_noise(
        request, window, dimensions.channels * latent_length, noise);
    result->forced_noise_used.push_back(forced ? 1 : 0);
    if (!mm3_sample_window(model, request, progress, noise, condition, carry, window,
                           window_count, latent_length, overlap, latents, result, error)) {
        return false;
    }
    mm3_update_window_carry(model, dimensions, latents, condition, latent_length, carry);
    return true;
}

static bool mm3_generate_windows(const MM3Model & model, const MM3GenRequest & request,
                                 const MM3PipelineDimensions & dimensions,
                                 const MM3ProgressCb & progress, MM3GenResult * result,
                                 const std::vector<int64_t> & starts,
                                 std::vector<std::vector<float>> & chunk_latents,
                                 std::string * error) {
    MM3WindowCarry carry;
    const int64_t window_count = static_cast<int64_t>(starts.size());
    chunk_latents.resize(static_cast<size_t>(window_count));
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_generate_window(model, request, dimensions, progress, result, starts,
                                 window, carry, chunk_latents[static_cast<size_t>(window)],
                                 error)) {
            return false;
        }
    }
    return true;
}

static bool mm3_vocode_windows(const MM3Model & model, const MM3GenRequest & request,
                               const MM3ProgressCb & progress, MM3GenResult * result,
                               const std::vector<std::vector<float>> & chunk_latents,
                               std::vector<std::vector<float>> & waveforms,
                               std::string * error) {
    const int64_t window_count = static_cast<int64_t>(chunk_latents.size());
    waveforms.resize(static_cast<size_t>(window_count));
    const auto started = std::chrono::steady_clock::now();
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_emit_progress(progress, {"vocode", window, window_count, 0, 1},
                               request.should_cancel, error)) {
            return false;
        }
        if (!mm3_vocoder_decode(model, chunk_latents[static_cast<size_t>(window)].data(),
                                result->window_L[static_cast<size_t>(window)],
                                waveforms[static_cast<size_t>(window)], error)) {
            return false;
        }
    }
    result->voc_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return true;
}

static bool mm3_calculate_crop_spans(const MM3GenRequest & request,
                                     const std::vector<int64_t> & latent_lengths,
                                     int64_t upsample, std::vector<int64_t> & left,
                                     std::vector<int64_t> & length, int64_t & total,
                                     std::string * error) {
    const int64_t window_count = static_cast<int64_t>(latent_lengths.size());
    left.resize(static_cast<size_t>(window_count));
    length.resize(static_cast<size_t>(window_count));
    total = 0;
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_continue_generation(request.should_cancel, error)) {
            return false;
        }
        const auto span = tts_cpp::minimax::detail::crop_span(
            latent_lengths[static_cast<size_t>(window)], window, window_count, upsample);
        left[static_cast<size_t>(window)] = span.left;
        length[static_cast<size_t>(window)] = span.length;
        total += span.length;
    }
    return true;
}

static void mm3_copy_stereo_window(const std::vector<float> & waveform, int64_t source_length,
                                   int64_t source_left, int64_t output_length,
                                   int64_t output_offset, int64_t copy_length,
                                   std::vector<float> & output) {
    for (int channel = 0; channel < kMm3StereoChannels; ++channel) {
        memcpy(output.data() + static_cast<int64_t>(channel) * output_length + output_offset,
               waveform.data() + static_cast<int64_t>(channel) * source_length + source_left,
               static_cast<size_t>(copy_length) * sizeof(float));
    }
}

static bool mm3_copy_audio_windows(const MM3GenRequest & request,
                                   const std::vector<int64_t> & latent_lengths,
                                   const std::vector<std::vector<float>> & waveforms,
                                   const std::vector<int64_t> & left,
                                   const std::vector<int64_t> & length, int64_t upsample,
                                   int64_t total, std::vector<float> & output,
                                   std::string * error) {
    int64_t output_offset = 0;
    const int64_t window_count = static_cast<int64_t>(waveforms.size());
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_continue_generation(request.should_cancel, error)) {
            return false;
        }
        const int64_t source_length =
            latent_lengths[static_cast<size_t>(window)] * upsample;
        mm3_copy_stereo_window(
            waveforms[static_cast<size_t>(window)], source_length,
            left[static_cast<size_t>(window)], total, output_offset,
            length[static_cast<size_t>(window)], output);
        output_offset += length[static_cast<size_t>(window)];
    }
    return true;
}

static bool mm3_sanitize_audio(const MM3GenRequest & request, MM3GenResult * result,
                               std::string * error) {
    double sum_squared = 0.0;
    for (size_t index = 0; index < result->audio.size(); ++index) {
        if (index % kMm3CancellationPollSamples == 0 &&
            !mm3_continue_generation(request.should_cancel, error)) {
            return false;
        }
        float & value = result->audio[index];
        if (!std::isfinite(value)) {
            result->has_nan = true;
            value = 0.0f;
            continue;
        }
        value = std::max(-1.0f, std::min(1.0f, value));
        result->peak = std::max(result->peak, std::fabs(value));
        sum_squared += static_cast<double>(value) * static_cast<double>(value);
    }
    result->rms = result->audio.empty()
                      ? 0.0
                      : std::sqrt(sum_squared / static_cast<double>(result->audio.size()));
    return true;
}

static bool mm3_stitch_audio(const MM3GenRequest & request,
                             const MM3PipelineDimensions & dimensions,
                             const MM3ProgressCb & progress,
                             const std::vector<std::vector<float>> & waveforms,
                             MM3GenResult * result, std::string * error) {
    const int64_t window_count = static_cast<int64_t>(waveforms.size());
    if (!mm3_emit_progress(progress, {"stitch", -1, window_count, 0, 1},
                           request.should_cancel, error)) {
        return false;
    }
    std::vector<int64_t> left;
    std::vector<int64_t> length;
    int64_t total = 0;
    if (!mm3_calculate_crop_spans(request, result->window_L, dimensions.upsample,
                                  left, length, total, error)) {
        return false;
    }
    result->audio.assign(
        static_cast<size_t>(kMm3StereoChannels * total), 0.0f);
    result->n_samples = total;
    if (!mm3_copy_audio_windows(request, result->window_L, waveforms, left, length,
                                dimensions.upsample, total, result->audio, error)) {
        return false;
    }
    return mm3_sanitize_audio(request, result, error);
}

static bool mm3_finish_generation(const MM3GenRequest & request,
                                  const MM3ProgressCb & progress,
                                  const std::chrono::steady_clock::time_point & started,
                                  std::vector<std::vector<float>> & chunk_latents,
                                  MM3GenResult * result, std::string * error) {
    if (request.keep_window_latents) {
        result->window_latents = std::move(chunk_latents);
    }
    result->total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (!mm3_emit_progress(progress,
                           {"done", -1, result->n_windows, result->n_windows, result->n_windows},
                           request.should_cancel, error)) {
        return false;
    }
    fprintf(stderr,
            "[MM3-Pipe] %lld frames -> %lld window(s) -> %lld samples/ch (%.2fs @ %d Hz) | "
            "AR %.0f ms, cond %.0f ms, flow %.0f ms, voc %.0f ms, total %.0f ms\n",
            static_cast<long long>(result->frames),
            static_cast<long long>(result->n_windows),
            static_cast<long long>(result->n_samples),
            static_cast<double>(result->n_samples) /
                static_cast<double>(result->sample_rate > 0 ? result->sample_rate : 1),
            result->sample_rate, result->ar_ms, result->cond_ms, result->flow_ms,
            result->voc_ms, result->total_ms);
    return true;
}

static bool mm3_generate(const MM3Model & model, const MM3GenRequest & request,
                         MM3Tokenizer * tokenizer, const MM3ProgressCb & progress,
                         MM3GenResult * result, std::string * error) {
    const auto started = std::chrono::steady_clock::now();
    const MM3PipelineDimensions dimensions = mm3_pipeline_dimensions(model);
    *result = MM3GenResult{};
    result->sample_rate = static_cast<int>(model.synth_cfg.voc.sampling_rate);
    std::vector<int32_t> conditional;
    std::vector<int32_t> unconditional;
    if (!mm3_prepare_token_ids(model, request, tokenizer, conditional, unconditional, error)) {
        return false;
    }
    if (!mm3_run_ar_stage(model, request, dimensions, conditional, unconditional,
                          progress, result, error)) {
        return false;
    }
    result->chunk_starts = tts_cpp::minimax::detail::window_starts(
        result->frames, dimensions.window_frames, dimensions.hop_frames);
    result->n_windows = static_cast<int64_t>(result->chunk_starts.size());
    std::vector<std::vector<float>> chunk_latents;
    if (!mm3_generate_windows(model, request, dimensions, progress, result,
                              result->chunk_starts, chunk_latents, error)) {
        return false;
    }
    std::vector<std::vector<float>> waveforms;
    if (!mm3_vocode_windows(model, request, progress, result, chunk_latents,
                            waveforms, error)) {
        return false;
    }
    if (!mm3_stitch_audio(request, dimensions, progress, waveforms, result, error)) {
        return false;
    }
    return mm3_finish_generation(request, progress, started, chunk_latents, result, error);
}
