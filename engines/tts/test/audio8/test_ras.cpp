// Repetition-aware sampling, which the fixture tests cannot reach: they decode
// greedily, and pick() skips the retry under greedy. So what counts as a repeat,
// what the retry narrows to, and what a window of zero means are only pinned
// here.
//
// The distributions are built so the retry is observable. two_way() puts a
// runner-up just under the leader, far enough above the tail that an ordinary
// draw reaches either, while the narrower nucleus the retry falls back to keeps
// the leader alone -- so a suppressed candidate cannot appear in the output.
// Each trial starts from a fresh sampler, because a window that has absorbed a
// few accepted tokens no longer holds the entry under test.

#include "audio8/sampling.h"

#include <cstdio>
#include <random>
#include <set>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

constexpr int SEMANTIC_BEGIN = 0;
constexpr int SEMANTIC_END = 3;
constexpr int EOS = 4;
constexpr int VOCAB = 5;
constexpr int TRIALS = 200;
constexpr int LEADER = 3;
constexpr int SEEDED = 0;
constexpr int WINDOW = 4;
constexpr float LEADER_LOGIT = 10.0f;
constexpr float RUNNER_UP_LOGIT = 9.9f;
constexpr float TAIL_LOGIT = 1.0f;
constexpr float PEAK_LOGIT = 40.0f;
constexpr float NARROW_TEMPERATURE = 0.5f;
constexpr float NARROW_TOP_P = 0.5f;

int failures = 0;

void check(bool condition, const char * what, int line) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", what, __FILE__, line);
}

#define CHECK(cond, msg) check((cond), (msg), __LINE__)

sampling_params wide() {
    sampling_params params;
    params.greedy = false;
    params.temperature = 1.0f;
    params.top_k = 0;
    params.top_p = 1.0f;
    return params;
}

sampling_params narrow() {
    sampling_params params = wide();
    params.temperature = NARROW_TEMPERATURE;
    params.top_p = NARROW_TOP_P;
    return params;
}

sampling_params greedy() {
    sampling_params params;
    params.greedy = true;
    return params;
}

RepetitionAwareSampler with_window(int window) {
    return RepetitionAwareSampler(window, SEMANTIC_BEGIN, SEMANTIC_END, narrow());
}

// LEADER takes just over half the mass and `runner_up` just under it: the
// narrow nucleus drops everything past the first rank, the wide one keeps both.
std::vector<float> two_way(int runner_up) {
    std::vector<float> logits(VOCAB, TAIL_LOGIT);
    logits[LEADER] = LEADER_LOGIT;
    logits[runner_up] = RUNNER_UP_LOGIT;
    return logits;
}

std::vector<float> peaked_at(int index) {
    std::vector<float> logits(VOCAB, 0.0f);
    logits[index] = PEAK_LOGIT;
    return logits;
}

// The opening pick is where remember() seeds the window with zeros; the token it
// chose is deliberately not recorded, so priming is what puts zeros behind us.
void prime(RepetitionAwareSampler & sampler, std::mt19937 & rng) {
    sampler.pick(peaked_at(LEADER), wide(), rng);
}

void prime_control(std::mt19937 & rng) {
    sample_token(peaked_at(LEADER), wide(), rng);
}

// One pick per seed, each on a sampler whose window holds nothing but zeros.
std::set<int> picks_after_priming(int window, int runner_up) {
    std::set<int> seen;
    for (int trial = 0; trial < TRIALS; ++trial) {
        RepetitionAwareSampler sampler = with_window(window);
        std::mt19937 rng(static_cast<uint32_t>(trial));
        prime(sampler, rng);
        seen.insert(sampler.pick(two_way(runner_up), wide(), rng));
    }
    return seen;
}

// The same seeds at the same point in the stream, without the window.
std::set<int> draws_after_priming(int runner_up) {
    std::set<int> seen;
    for (int trial = 0; trial < TRIALS; ++trial) {
        std::mt19937 rng(static_cast<uint32_t>(trial));
        prime_control(rng);
        seen.insert(sample_token(two_way(runner_up), wide(), rng));
    }
    return seen;
}

bool picks_stay(RepetitionAwareSampler & sampler, const std::vector<float> & logits,
                const sampling_params & params, std::mt19937 & rng, int expected) {
    for (int trial = 0; trial < TRIALS; ++trial) {
        if (sampler.pick(logits, params, rng) != expected) return false;
    }
    return true;
}

// Whether `token` ever comes out twice running, which is what a token the
// window is allowed to hold can do and a repeat cannot.
bool repeats_consecutively(RepetitionAwareSampler & sampler,
                           const std::vector<float> & logits, std::mt19937 & rng,
                           int token) {
    int previous = -1;
    for (int trial = 0; trial < TRIALS; ++trial) {
        const int chosen = sampler.pick(logits, wide(), rng);
        if (chosen == token && previous == token) return true;
        previous = chosen;
    }
    return false;
}

// A window of zero used to leave the history empty and then pop it anyway. The
// tokens that come back are the same either way, so this only fails where the
// standard library checks the precondition -- hence the _GLIBCXX_ASSERTIONS the
// target is built with.
void test_zero_window_survives_repeated_picks() {
    RepetitionAwareSampler sampler = with_window(0);
    std::mt19937 rng(1);
    CHECK(picks_stay(sampler, peaked_at(LEADER), wide(), rng, LEADER),
          "window 0: every pick is the dominant candidate");
}

// With no window there is nothing to compare against, so the draw has to come
// through untouched -- including how much of the generator it consumed, which a
// diverging sequence would expose.
void test_zero_window_leaves_the_draw_alone() {
    RepetitionAwareSampler sampler = with_window(0);
    std::mt19937 picked(2);
    std::mt19937 drawn(2);
    const std::vector<float> logits = two_way(SEEDED);
    bool same = true;
    for (int trial = 0; trial < TRIALS; ++trial) {
        if (sampler.pick(logits, wide(), picked) != sample_token(logits, wide(), drawn)) {
            same = false;
        }
    }
    CHECK(same, "window 0: the draw is exactly what sample_token would return");
}

// The window is seeded with zeros, so codebook index 0 is a repeat on the pick
// after the first and the retry stands in for it.
void test_seeded_window_suppresses_its_token() {
    const std::set<int> picked = picks_after_priming(WINDOW, SEEDED);
    CHECK(picked.count(SEEDED) == 0, "a token in the window never comes out");
    CHECK(picked.count(LEADER) == 1, "the narrow retry lands on the leader");
    CHECK(draws_after_priming(SEEDED).count(SEEDED) == 1,
          "the same seeds do reach that token without the window");
}

// EOS is outside [semantic_begin, semantic_end], so it is never a repeat however
// often it lands: an end of speech must not be second-guessed. Consecutive draws
// are what separates it from a semantic token, since being in the window is only
// visible on the pick after.
void test_eos_repeats_freely() {
    RepetitionAwareSampler sampler = with_window(WINDOW);
    std::mt19937 rng(4);
    prime(sampler, rng);
    CHECK(repeats_consecutively(sampler, two_way(EOS), rng, EOS),
          "EOS follows EOS");
}

void test_a_semantic_token_does_not_repeat_freely() {
    RepetitionAwareSampler sampler = with_window(WINDOW);
    std::mt19937 rng(6);
    prime(sampler, rng);
    CHECK(!repeats_consecutively(sampler, two_way(SEMANTIC_END - 2), rng,
                                 SEMANTIC_END - 2),
          "a semantic token in the same position never follows itself");
}

void test_greedy_ignores_the_window() {
    RepetitionAwareSampler sampler = with_window(WINDOW);
    std::mt19937 rng(5);
    prime(sampler, rng);
    CHECK(picks_stay(sampler, peaked_at(SEEDED), greedy(), rng, SEEDED),
          "greedy: the argmax is taken even when it repeats");
}

// The window holds `window` entries and evicts oldest-first, so a one-entry
// window loses its seeded zero as soon as a real token is recorded.
void test_window_evicts_oldest_first() {
    std::set<int> seen;
    for (int trial = 0; trial < TRIALS; ++trial) {
        RepetitionAwareSampler sampler = with_window(1);
        std::mt19937 rng(static_cast<uint32_t>(trial));
        prime(sampler, rng);
        sampler.pick(two_way(SEEDED), wide(), rng);
        seen.insert(sampler.pick(two_way(SEEDED), wide(), rng));
    }
    CHECK(seen.count(SEEDED) == 1, "an evicted token is drawable again");
}

}  // namespace

int main() {
    test_zero_window_survives_repeated_picks();
    test_zero_window_leaves_the_draw_alone();
    test_seeded_window_suppresses_its_token();
    test_eos_repeats_freely();
    test_a_semantic_token_does_not_repeat_freely();
    test_greedy_ignores_the_window();
    test_window_evicts_oldest_first();

    if (failures == 0) {
        std::fprintf(stderr, "audio8 ras: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "audio8 ras: %d failure(s)\n", failures);
    return 1;
}
