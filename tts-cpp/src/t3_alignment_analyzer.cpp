#include "t3_alignment_analyzer.h"

#include <algorithm>

namespace tts_cpp::chatterbox::detail {

void t3_alignment_analyzer::reset(const t3_align_analyzer_params & p) {
    p_             = p;
    frame_         = 0;
    text_position_ = 0;
    complete_      = false;
    completed_at_  = -1;
    tail_sum_.assign((size_t) std::max(0, p_.tail_cols), 0.0);
    early_sum_     = 0.0;
    recent_tokens_.clear();
}

t3_align_action t3_alignment_analyzer::step(const std::vector<float> & row,
                                            int32_t sampled_token) {
    // Track tokens regardless so the repetition check has history even on the
    // frames where the alignment row is unavailable.
    recent_tokens_.push_back(sampled_token);
    if (recent_tokens_.size() > 8) {
        recent_tokens_.erase(recent_tokens_.begin(),
                             recent_tokens_.end() - 8);
    }

    const int S = p_.text_len;
    if (!p_.enabled || S < p_.min_text_len || row.empty()) {
        return t3_align_action::none;
    }

    const int n = std::min((int) row.size(), S);
    if (n <= 0) return t3_align_action::none;

    // Monotonic mask: a frame cannot align to text it has not reached yet, so
    // zero out columns beyond curr_frame_pos + 1 (curr_frame_pos == frame_).
    // This is also the safeguard against a start-of-speech hallucination
    // spiking on a late text column and triggering an immediate false
    // "complete": since cur <= frame_ + 1, `complete_` (text_position >= S -
    // complete_margin) is impossible before frame (S - complete_margin - 1).
    std::vector<float> a(row.begin(), row.begin() + n);
    for (int c = frame_ + 1; c < n; ++c) a[(size_t) c] = 0.0f;

    // Current text position = argmax of the masked row.
    int cur = 0;
    float mv = a[0];
    for (int c = 1; c < n; ++c) {
        if (a[(size_t) c] > mv) { mv = a[(size_t) c]; cur = c; }
    }

    // Position update with a discontinuity guard (-disc_back < delta < disc_fwd).
    const int delta = cur - text_position_;
    const bool discontinuity = !(delta > -p_.disc_back && delta < p_.disc_fwd);
    if (!discontinuity) text_position_ = cur;

    // --- completion --------------------------------------------------------
    if (!complete_ && text_position_ >= S - p_.complete_margin) {
        complete_ = true;
        completed_at_ = frame_ + 1;   // post-completion window starts next frame
    }

    // --- post-completion running reductions --------------------------------
    if (complete_ && frame_ >= completed_at_) {
        const int tc = std::max(0, p_.tail_cols);
        const int tail_from = std::max(0, n - tc);
        for (int c = tail_from; c < n; ++c) {
            const int idx = c - tail_from;
            if (idx >= 0 && idx < (int) tail_sum_.size())
                tail_sum_[(size_t) idx] += a[(size_t) c];
        }
        const int early_to = std::max(0, n - p_.rep_back_cols);
        float early_max = 0.0f;
        for (int c = 0; c < early_to; ++c) early_max = std::max(early_max, a[(size_t) c]);
        early_sum_ += early_max;
    }

    // --- decision ----------------------------------------------------------
    double tail_max = 0.0;
    for (double v : tail_sum_) tail_max = std::max(tail_max, v);
    const bool long_tail = complete_ && (tail_max >= (double) p_.long_tail_thresh);
    const bool alignment_repetition = complete_ && (early_sum_ > (double) p_.rep_thresh);

    // Token repetition (2x identical), gated behind completion to avoid the
    // reference's known mid-text early-cutoff (issues #519/#587).
    bool token_repetition = false;
    if (recent_tokens_.size() >= 3) {
        const size_t m = recent_tokens_.size();
        token_repetition = (recent_tokens_[m - 1] == recent_tokens_[m - 2]);
    }
    token_repetition = token_repetition && complete_;

    ++frame_;

    if (long_tail || alignment_repetition || token_repetition) {
        return t3_align_action::force_eos;
    }
    // Not yet near the end: tell the caller to hold off on an early stop.
    if (p_.suppress_eos_enabled && cur < S - p_.complete_margin && S > p_.min_text_len) {
        return t3_align_action::suppress_eos;
    }
    return t3_align_action::none;
}

} // namespace tts_cpp::chatterbox::detail
