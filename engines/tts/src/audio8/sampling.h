#pragma once
// Audio8 token sampling, including the repetition-aware step the reference
// calls RAS.
//
// The filter order is the reference's and is load-bearing: top-k and top-p run
// on the raw logits, and only what survives is divided by the temperature.
// Doing it the other way round -- the order most samplers use -- changes which
// candidates the nucleus keeps.

#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {

struct sampling_params {
    bool greedy = false;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
};

// The scores the draw sees: everything top-k or top-p rejected set to -inf,
// the rest scaled by the temperature.
std::vector<float> filter_scores(const std::vector<float> & logits,
                                 const sampling_params & params);

// Draws one index from `logits`, which the caller has already masked.
int sample_token(const std::vector<float> & logits, const sampling_params & params,
                 std::mt19937 & rng);

// Repetition-aware sampling: when the ordinary draw repeats a semantic token
// from the recent window, the reference re-draws from a narrower nucleus at a
// higher temperature instead of accepting the repeat. Only semantic tokens are
// eligible, so an end-of-speech draw is never second-guessed.
class RepetitionAwareSampler {
public:
    RepetitionAwareSampler(int window, int semantic_begin, int semantic_end,
                           const sampling_params & narrow);

    int pick(const std::vector<float> & logits, const sampling_params & params,
             std::mt19937 & rng);

private:
    int window_ = 0;
    int semantic_begin_ = 0;
    int semantic_end_ = 0;
    sampling_params narrow_;
    // Empty until the second step: the reference starts the window as zeros
    // after the first draw, so the opening token is never treated as history.
    // A window of zero keeps it empty and turns the resampling off.
    std::deque<int> recent_;
    bool started_ = false;

    bool is_repeat(int token) const;
    void start_window();
    void record(int token);
    void remember(int token);
};

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
