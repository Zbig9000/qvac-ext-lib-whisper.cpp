#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace tts_cpp {
namespace parler {
namespace detail {

namespace {

int32_t argmax_row(const float * row, int vocab) {
    int best = 0;
    for (int i = 1; i < vocab; ++i) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

void scale_row(const float * row, int vocab, float temperature,
               std::vector<float> & scaled) {
    scaled.resize((size_t) vocab);
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < vocab; ++i) scaled[i] = row[i] / temperature;
    } else {
        std::copy(row, row + vocab, scaled.begin());
    }
}

float top_k_threshold(const std::vector<float> & scaled, int top_k,
                      std::vector<float> & work) {
    work = scaled;
    std::nth_element(work.begin(), work.begin() + (top_k - 1), work.end(),
                     std::greater<float>());
    return work[top_k - 1];
}

// Every id at or above the threshold survives, so threshold ties keep more
// than top_k candidates -- exactly like masking below the nth_element value.
void collect_candidates(const std::vector<float> & scaled, float thresh,
                        std::vector<int> & cand) {
    cand.clear();
    for (int i = 0; i < (int) scaled.size(); ++i) {
        if (scaled[i] >= thresh) cand.push_back(i);
    }
}

// HF top-p: keep the smallest descending prefix whose mass exceeds top_p.
// Restores ascending id order afterwards so the multinomial walk below sums
// probabilities in the same order as a full-vocabulary pass.
void shrink_to_nucleus(const std::vector<float> & scaled, float top_p,
                       std::vector<int> & cand, std::vector<double> & probs) {
    std::sort(cand.begin(), cand.end(),
              [&](int a, int b) { return scaled[a] > scaled[b]; });
    const float max_l = scaled[cand[0]];
    probs.resize(cand.size());
    double denom = 0.0;
    for (size_t j = 0; j < cand.size(); ++j) {
        probs[j] = std::exp((double) scaled[cand[j]] - max_l);
        denom += probs[j];
    }
    size_t keep = cand.size();
    double cum = 0.0;
    for (size_t j = 0; j < cand.size(); ++j) {
        cum += probs[j] / denom;
        if (cum > top_p) {
            keep = j + 1;
            break;
        }
    }
    cand.resize(keep);
    std::sort(cand.begin(), cand.end());
}

// softmax + multinomial over the candidates, one uniform draw per row.
int32_t draw_from_candidates(const std::vector<float> & scaled,
                             const std::vector<int> & cand,
                             std::vector<double> & probs, std::mt19937 & rng) {
    float max_l = -std::numeric_limits<float>::infinity();
    for (int i : cand) max_l = std::max(max_l, scaled[i]);
    probs.assign(cand.size(), 0.0);
    double denom = 0.0;
    for (size_t j = 0; j < cand.size(); ++j) {
        const float v = scaled[cand[j]];
        if (std::isinf(v) && v < 0) continue;
        probs[j] = std::exp((double) v - max_l);
        denom += probs[j];
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double r = dist(rng) * denom;
    double cum = 0.0;
    for (size_t j = 0; j < cand.size(); ++j) {
        cum += probs[j];
        if (r <= cum) return cand[j];
    }
    return cand.back();
}

int32_t sample_row(const float * row, int vocab, const parler_sampling_params & p,
                   std::mt19937 & rng, parler_sampler_scratch & s) {
    if (p.greedy) return argmax_row(row, vocab);

    scale_row(row, vocab, p.temperature, s.scaled);
    if (p.top_k > 0 && p.top_k < vocab) {
        collect_candidates(s.scaled, top_k_threshold(s.scaled, p.top_k, s.work), s.cand);
    } else {
        collect_candidates(s.scaled, -std::numeric_limits<float>::infinity(), s.cand);
    }
    if (p.top_p < 1.0f) {
        shrink_to_nucleus(s.scaled, p.top_p, s.cand, s.probs);
    }
    return draw_from_candidates(s.scaled, s.cand, s.probs, rng);
}

} // namespace

parler_sampling_params parler_resolve_sampling(const parler_sampling_request & req,
                                               const parler_gen_defaults & def,
                                               std::string * repaired) {
    parler_sampling_params p;
    p.temperature = req.temperature > 0.0f ? req.temperature : def.temperature;
    p.top_k       = req.top_k > 0 ? req.top_k : def.top_k;
    // A small top_p also narrows toward the argmax, but only on peaked steps,
    // so it stays a genuine nucleus knob rather than a repaired one.
    p.top_p       = req.top_p;

    // top_k == 1 masks every logit below the largest, leaving the multinomial
    // draw a single candidate -- identical to greedy.
    const char * reason = req.greedy      ? "greedy = true"
                        : p.top_k == 1    ? "top_k = 1"
                        : !def.do_sample  ? "the GGUF's do_sample = false"
                                          : nullptr;
    p.greedy = false;
    if (p.top_k == 1) {
        p.top_k = def.top_k > 1 ? def.top_k : 0;  // 0 => no top-k filter
    }
    if (repaired) {
        if (reason) *repaired = reason;
        else        repaired->clear();
    }
    return p;
}

std::vector<int32_t> parler_sample_frame(const float * logits, int n_codebooks, int vocab,
                                         const parler_sampling_params & params,
                                         std::mt19937 & rng,
                                         parler_sampler_scratch & scratch) {
    std::vector<int32_t> frame((size_t) n_codebooks);
    for (int k = 0; k < n_codebooks; ++k) {
        frame[k] = sample_row(logits + (size_t) k * vocab, vocab, params, rng, scratch);
    }
    return frame;
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
