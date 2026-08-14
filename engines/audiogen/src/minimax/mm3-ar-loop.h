#pragma once

#include "mm3-depth-graph.h"
#include "mm3-lm-graph.h"
#include "mm3-model.h"
#include "mm3-sample.h"
#include "mm3-tokenizer.h"
#include "logic.h"
#include "progress.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

struct MM3ArDump {
    std::vector<float> last_hidden;
    std::vector<float> sem_logits;
    std::vector<float> guided;
    std::vector<float> feedback;
    std::vector<float> depth_hidden;
};

struct MM3ArResult {
    int64_t n_iterations = 0;
    int64_t n_frames     = 0;
    int64_t hidden_dim   = 0;
    int64_t n_codebooks  = 0;
    int64_t sem_vocab    = 0;
    bool    eos_hit      = false;

    std::vector<int32_t> semantic_all;
    std::vector<int32_t> acoustic_all;
    std::vector<float>   frame_hiddens;
    std::vector<float>   prefill_hidden;
    std::vector<MM3ArDump> dumps;

    int64_t nonfinite_logits = 0;
    double  prefill_ms       = 0.0;
    double  lm_ms            = 0.0;
    double  depth_ms         = 0.0;
    double  host_ms          = 0.0;
    double  total_ms         = 0.0;
    int64_t lm_steps         = 0;
};

struct MM3ArOptions {
    int64_t  max_frames = 300;
    uint64_t seed       = 42;

    const int32_t * forced_semantic = nullptr;
    const int32_t * forced_acoustic = nullptr;
    int64_t         forced_len      = 0;

    int64_t dump_iters      = 0;
    bool    collect_hiddens = true;

    std::function<void(int64_t , int64_t )> on_frame;

    std::function<bool()> should_cancel;
};

static MM3LmGraph g_mm3_lm;

static bool mm3_ar_cancelled(const MM3ArOptions & options, std::string * error) {
    if (!tts_cpp::minimax::detail::cancellation_requested(options.should_cancel)) {
        return false;
    }
    if (error) {
        *error = MM3_ERR_CANCELLED;
    }
    return true;
}

static bool mm3_ar_plan(const MM3Model & m, const int32_t * cond_ids, const int32_t * uncond_ids, int64_t n_prompt,
                        const MM3ArOptions & opt, MM3ArResult * out, std::string * err) {
    const MM3LmConfig & c  = m.lm_cfg;
    const int64_t       H  = (int64_t) c.embedding_length;
    const int64_t       V  = (int64_t) c.vocab_size;
    const int64_t       SV = (int64_t) c.semantic_vocab_size;
    const int64_t       NC = (int64_t) c.num_codebooks - 1;
    const int64_t       AV = (int64_t) c.acoustic_vocab_size;
    const int64_t       OFF  = (int64_t) c.semantic_vocab_offset;
    const int64_t       EOS  = (int64_t) c.eos_audio;
    const float         CFG  = c.ar_cfg_scale > 0.0f ? c.ar_cfg_scale : 1.5f;
    const int           TOPK = c.ar_top_k > 0 ? (int) c.ar_top_k : 50;

    if (n_prompt <= 0) {
        if (err) {
            *err = "the prompt tokenised to zero tokens";
        }
        return false;
    }
    if (c.max_prompt_tokens > 0 && n_prompt > (int64_t) c.max_prompt_tokens) {
        if (err) {
            *err = "the prompt is " + std::to_string((long long) n_prompt) + " tokens; the checkpoint's limit is " +
                   std::to_string(c.max_prompt_tokens);
        }
        return false;
    }
    if (EOS < 0 || EOS >= V || OFF + SV > V) {
        if (err) {
            *err = "the LM vocabulary metadata does not cover the semantic range and EOS";
        }
        return false;
    }

    int64_t max_frames = opt.max_frames;
    if (max_frames <= 0) {
        if (err) {
            *err = "max_frames must be positive";
        }
        return false;
    }
    if (c.max_audio_frames > 0 && max_frames > (int64_t) c.max_audio_frames) {
        max_frames = (int64_t) c.max_audio_frames;
    }
    const bool forced = opt.forced_semantic != nullptr;
    if (forced) {
        if (!opt.forced_acoustic || opt.forced_len <= 0) {
            if (err) {
                *err = "forced replay needs both forced_semantic and forced_acoustic, and a positive length";
            }
            return false;
        }
        if (opt.forced_len - 1 < max_frames) {
            max_frames = opt.forced_len - 1;
        }
        if (max_frames <= 0) {
            if (err) {
                *err = "forced replay needs at least 2 iterations (one un-emitted, one emitted)";
            }
            return false;
        }
    }

    if (!mm3_lm_prepare(m, &g_mm3_lm, n_prompt + max_frames + 2, err)) {
        return false;
    }

    *out             = MM3ArResult{};
    out->hidden_dim  = H;
    out->n_codebooks = NC + 1;
    out->sem_vocab   = SV;

    std::vector<float> hidden((size_t) (H * MM3_LM_CFG_ROWS));
    std::vector<float> logits((size_t) (V * MM3_LM_CFG_ROWS));
    std::vector<float> feedback((size_t) H);

    const auto t_start = std::chrono::steady_clock::now();
    {
        if (mm3_ar_cancelled(opt, err)) {
            return false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (!mm3_lm_prefill(m, &g_mm3_lm, cond_ids, uncond_ids, n_prompt, hidden.data(), logits.data(), err)) {
            return false;
        }
        out->prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    out->prefill_hidden.assign(hidden.begin(), hidden.end());

    const int64_t      NCAND = SV + 1;
    std::vector<float> cand_cond((size_t) NCAND);
    std::vector<float> cand_unc((size_t) NCAND);
    std::vector<float> cand_guided((size_t) NCAND);
    std::vector<float> sel_scratch;
    std::vector<float> samp_scratch;

    std::mt19937_64      rng(opt.seed);
    std::vector<int32_t> ac_rows((size_t) NC);
    MM3DepthFrame        frame;

    out->semantic_all.reserve((size_t) (max_frames + 1));
    out->acoustic_all.reserve((size_t) ((max_frames + 1) * NC));
    if (opt.collect_hiddens) {
        out->frame_hiddens.reserve((size_t) (max_frames * (NC + 1) * H));
    }

    for (int64_t it = 0; it <= max_frames; it++) {
        if (mm3_ar_cancelled(opt, err)) {
            return false;
        }
        if (forced && it >= opt.forced_len) {
            break;
        }
        const auto t_host0 = std::chrono::steady_clock::now();

        const float * lrow_c = logits.data();
        const float * lrow_u = logits.data() + V;
        auto          fix    = [&](float x) -> float {
            if (std::isnan(x) || (std::isinf(x) && x > 0.0f)) {
                out->nonfinite_logits++;
                return -INFINITY;
            }
            return x;
        };
        cand_cond[0] = fix(lrow_c[EOS]);
        cand_unc[0]  = fix(lrow_u[EOS]);
        for (int64_t j = 0; j < SV; j++) {
            cand_cond[(size_t) (j + 1)] = fix(lrow_c[OFF + j]);
            cand_unc[(size_t) (j + 1)]  = fix(lrow_u[OFF + j]);
        }

        for (int64_t i = 0; i < NCAND; i++) {
            const float u          = cand_unc[(size_t) i];
            cand_guided[(size_t) i] = u + (cand_cond[(size_t) i] - u) * CFG;
        }
        {
            int64_t k = TOPK < NCAND ? (int64_t) TOPK : NCAND;
            if (k < 1) {
                k = 1;
            }
            float threshold = -INFINITY;
            if (k < NCAND) {
                sel_scratch = cand_cond;
                std::nth_element(sel_scratch.begin(), sel_scratch.begin() + (size_t) (k - 1), sel_scratch.end(),
                                 std::greater<float>());
                threshold = sel_scratch[(size_t) (k - 1)];
            }
            for (int64_t i = 0; i < NCAND; i++) {

                if (cand_cond[(size_t) i] < threshold) {
                    cand_guided[(size_t) i] = -INFINITY;
                }
            }
        }

        int32_t semantic;
        if (forced) {
            semantic = opt.forced_semantic[it];
            if (semantic < 0 || (int64_t) semantic >= SV) {
                if (err) {
                    *err = "forced semantic code " + std::to_string(semantic) + " at iteration " +
                           std::to_string((long long) it) + " is outside [0, " + std::to_string((long long) SV) + ")";
                }
                return false;
            }
        } else {
            const int64_t idx = mm3_sample_top_k(cand_guided.data(), NCAND, TOPK, rng, &samp_scratch);
            if (idx == 0) {
                out->eos_hit = true;
                out->host_ms +=
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();
                break;
            }
            semantic = (int32_t) (idx - 1);
        }
        out->host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();

        const int32_t * forced_ac = forced ? opt.forced_acoustic + it * NC : nullptr;
        if (!mm3_depth_decode_frame(m, hidden.data(), hidden.data() + H, semantic, forced_ac, &frame, err,
                                    forced ? nullptr : &rng, TOPK)) {
            return false;
        }
        if (mm3_ar_cancelled(opt, err)) {
            return false;
        }
        out->depth_ms += frame.ms;
        if (frame.n_codes != (int) NC) {
            if (err) {
                *err = "the depth decoder returned " + std::to_string(frame.n_codes) + " codes, expected " +
                       std::to_string((long long) NC);
            }
            return false;
        }

        out->semantic_all.push_back(semantic);
        for (int64_t i = 0; i < NC; i++) {
            out->acoustic_all.push_back(frame.codes[i]);
        }
        out->n_iterations++;

        const bool dumping = (int64_t) out->dumps.size() < opt.dump_iters;
        if (dumping) {
            MM3ArDump d;
            d.last_hidden.assign(hidden.begin(), hidden.end());
            d.sem_logits.resize((size_t) (2 * SV));
            memcpy(d.sem_logits.data(), lrow_c + OFF, (size_t) SV * sizeof(float));
            memcpy(d.sem_logits.data() + SV, lrow_u + OFF, (size_t) SV * sizeof(float));
            d.guided.assign(cand_guided.begin() + 1, cand_guided.end());
            d.feedback.assign((size_t) (2 * H), 0.0f);
            d.depth_hidden = frame.hiddens;
            out->dumps.push_back(std::move(d));
        }

        if (it > 0) {
            if (opt.collect_hiddens) {
                out->frame_hiddens.insert(out->frame_hiddens.end(), hidden.begin(), hidden.begin() + H);
                out->frame_hiddens.insert(out->frame_hiddens.end(), frame.hiddens.begin(), frame.hiddens.end());
            }
            out->n_frames++;
            if (opt.on_frame) {
                opt.on_frame(out->n_frames, max_frames);
            }
            if (mm3_ar_cancelled(opt, err)) {
                return false;
            }
            if (out->n_frames >= max_frames) {
                break;
            }
        }

        for (int64_t i = 0; i < NC; i++) {
            ac_rows[(size_t) i] = frame.codes[i] + (int32_t) (i * AV);
        }
        if (mm3_ar_cancelled(opt, err)) {
            return false;
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (!mm3_lm_decode(m, &g_mm3_lm, semantic + (int32_t) OFF, ac_rows.data(), hidden.data(), logits.data(),
                               feedback.data(), err)) {
                return false;
            }
            out->lm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            out->lm_steps++;
        }
        if (dumping) {
            MM3ArDump & d = out->dumps.back();

            memcpy(d.feedback.data(), feedback.data(), (size_t) H * sizeof(float));
            memcpy(d.feedback.data() + H, feedback.data(), (size_t) H * sizeof(float));
        }
    }

    out->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();

    if (out->n_frames == 0) {
        if (err) {
            *err = out->eos_hit ? "the LM emitted EOS on the first iteration; zero audio frames were generated"
                                : "zero audio frames were generated";
        }
        return false;
    }

    fprintf(stderr,
            "[MM3-AR] %lld frames (%lld iterations%s) in %.0f ms — prefill %.0f, LM %.0f (%lld steps, %.1f ms/step), "
            "depth %.0f (%.1f ms/frame), host %.0f\n",
            (long long) out->n_frames, (long long) out->n_iterations, out->eos_hit ? ", EOS" : "", out->total_ms,
            out->prefill_ms, out->lm_ms, (long long) out->lm_steps,
            out->lm_steps ? out->lm_ms / (double) out->lm_steps : 0.0, out->depth_ms,
            out->n_iterations ? out->depth_ms / (double) out->n_iterations : 0.0, out->host_ms);
    if (out->nonfinite_logits) {
        fprintf(stderr, "[MM3-AR] WARNING: %lld non-finite candidate logits were clamped to -inf\n",
                (long long) out->nonfinite_logits);
    }
    return true;
}
