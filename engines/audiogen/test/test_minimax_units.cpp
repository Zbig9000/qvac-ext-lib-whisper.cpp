#include "minimax/logic.h"
#include "minimax/bpe.h"
#include "minimax/progress.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

#define CHECK(condition)                                                                      \
    do {                                                                                      \
        ++checks;                                                                             \
        if (!(condition)) {                                                                   \
            ++failures;                                                                       \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition);          \
        }                                                                                     \
    } while (0)

template <typename Function>
bool throws_invalid_argument(Function function) {
    try {
        function();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

bool close(float left, float right, float tolerance = 1e-6f) {
    return std::fabs(left - right) <= tolerance;
}

tts_cpp::minimax::detail::ModelCompatibility valid_compatibility() {
    using namespace tts_cpp::minimax::detail;
    ModelCompatibility model;
    model.lm_embedding = 4096;
    model.lm_codebooks = 8;
    model.lm_acoustic_vocab = 1024;
    model.frame_rate = 25;
    model.max_audio_frames = 9000;
    model.max_prompt_tokens = 5000;
    model.depth_embedding = 4096;
    model.depth_codebooks = 8;
    model.depth_acoustic_vocab = 1024;
    model.condition_layers = 8;
    model.condition_hidden = 4096;
    model.condition_out = 2048;
    model.condition_rate = {24000, 960, 44100, 512};
    model.dit_condition = 2048;
    model.dit_channels = 128;
    model.window_frames = 200;
    model.hop_frames = 100;
    model.window_latents = 689;
    model.hop_latents = 344;
    model.vocoder_latent_channels = 128;
    model.vocoder_sampling_rate = 44100;
    model.vocoder_channels = 2;
    model.vocoder_upsample = 512;
    model.components = {"depth", "cond", "dit", "vocoder"};
    return model;
}

void touch(const std::filesystem::path & path) {
    std::ofstream stream(path);
    stream << "test";
}

void test_frame_validation() {
    using namespace tts_cpp::minimax::detail;
    CHECK(kDefaultFrameRate == 25);
    CHECK(kDefaultMaxFrames == 9000);
    CHECK(frames_from_duration(12.0, 25, 9000) == 300);
    CHECK(frames_from_duration(1000.0, 25, 9000) == 9000);
    CHECK(validate_frames(1, 9000) == 1);
    CHECK(validate_frames(9001, 9000) == 9000);
    CHECK(throws_invalid_argument([] { frames_from_duration(0.0, 25, 9000); }));
    CHECK(throws_invalid_argument([] { frames_from_duration(std::numeric_limits<double>::infinity(), 25, 9000); }));
    CHECK(throws_invalid_argument([] { validate_frames(0, 9000); }));
}

void test_prompt() {
    using tts_cpp::minimax::detail::build_prompt;
    const std::string prompt = build_prompt("## **Bright** pop", "[Verse] ignored\nHello ^ world");
    CHECK(prompt == "<|im_start|><|caption_start|>Bright pop<|caption_end|><|lyrics_start|>"
                    "[start]\n[verse]\nHello\nworld<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(build_prompt("Instrumental piano", "") ==
          "<|im_start|><|caption_start|>Instrumental piano<|caption_end|><|lyrics_start|>"
          "[start]\n[instrumental]<|lyrics_end|><|im_end|><|audio_start|>");
    CHECK(throws_invalid_argument([] { build_prompt(" \n\t", "words"); }));
}

void test_unconditional_mask() {
    using tts_cpp::minimax::detail::mask_unconditional;
    const std::vector<int32_t> conditional = {10, 11, 12, 13, 14, 15};
    CHECK(mask_unconditional(conditional, 99) == std::vector<int32_t>({10, 99, 99, 99, 14, 15}));
    CHECK(mask_unconditional({1, 2, 3}, 99) == std::vector<int32_t>({1, 2, 3}));
}

void test_noise() {
    using tts_cpp::minimax::detail::fill_noise;
    std::vector<float> first;
    std::vector<float> second;
    std::vector<float> other;
    fill_noise(42, 0, first, 8);
    fill_noise(42, 0, second, 8);
    fill_noise(42, 1, other, 8);
    CHECK(first == second);
    CHECK(first != other);
    CHECK(close(first[0], -0.19663458f));
    CHECK(close(first[1], -0.25129682f));
    CHECK(throws_invalid_argument([&] { fill_noise(42, -1, first, 8); }));
}

void test_flow_schedule() {
    using tts_cpp::minimax::detail::flow_schedule;
    CHECK(tts_cpp::minimax::detail::kDefaultFlowSteps == 30);
    CHECK(close(tts_cpp::minimax::detail::kDefaultCfgScale, 1.7f));
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    flow_schedule(4, sigmas, timesteps);
    CHECK(sigmas.size() == 5);
    CHECK(timesteps.size() == 4);
    CHECK(close(timesteps[0], 0.0f));
    CHECK(close(timesteps[1], 0.25f));
    CHECK(close(timesteps[2], 0.5f));
    CHECK(close(timesteps[3], 0.75f));
    CHECK(close(sigmas[4], 1.0f));
    CHECK(throws_invalid_argument([&] { flow_schedule(0, sigmas, timesteps); }));
}

void test_condition_length() {
    using namespace tts_cpp::minimax::detail;
    ConditionRate rate;
    CHECK(condition_latent_length(rate, 200) == 689);
    CHECK(condition_latent_length(rate, 100) == 344);
    CHECK(condition_latent_length(rate, 1) == 3);
    CHECK(throws_invalid_argument([&] { condition_latent_length(rate, 0); }));
}

void test_window_arithmetic() {
    using namespace tts_cpp::minimax::detail;
    CHECK(window_starts(200, kWindowFrames, kHopFrames) == std::vector<int64_t>({0}));
    CHECK(window_starts(300, kWindowFrames, kHopFrames) == std::vector<int64_t>({0, 100}));
    CHECK(window_starts(301, kWindowFrames, kHopFrames) == std::vector<int64_t>({0, 100, 200}));
    const CropSpan first = crop_span(689, 0, 2, 512);
    const CropSpan second = crop_span(689, 1, 2, 512);
    CHECK(first.left == 0);
    CHECK(first.length == 431 * 512);
    CHECK(second.left == 86 * 512);
    CHECK(second.length == 603 * 512);
    CHECK(stitched_sample_count({689, 689}, 512) == 529408);
    CHECK(kCarryLatents == kCropLeftLatents + kCropRightLatents);
    CHECK(kCarryLatents == 2 * kBlendLatents);
}

void test_sampling_edges() {
    using tts_cpp::minimax::detail::sample_top_k;
    std::mt19937_64 random(42);
    CHECK(sample_top_k(nullptr, 0, 50, random) == 0);
    const float logits[] = {-5.0f, 3.0f, 2.0f};
    CHECK(sample_top_k(logits, 3, 1, random) == 1);
    const float unusual[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
    CHECK(sample_top_k(unusual, 3, 1, random) == 1);
    std::mt19937_64 first(7);
    std::mt19937_64 second(7);
    CHECK(sample_top_k(logits, 3, 3, first) == sample_top_k(logits, 3, 3, second));
}

void test_model_compatibility() {
    using namespace tts_cpp::minimax::detail;
    ModelCompatibility model = valid_compatibility();
    CHECK(validate_model_compatibility(model).empty());
    model.condition_out = 1024;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.dit_channels = 64;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.depth_codebooks = 7;
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.components.push_back("vocoder");
    CHECK(!validate_model_compatibility(model).empty());
    model = valid_compatibility();
    model.condition_rate.output_sampling_rate = 32000;
    CHECK(!validate_model_compatibility(model).empty());
}

void test_unicode_categories() {
    CHECK(is_digit(0x0661));
    CHECK(is_letter(0x4E2D));
    CHECK(is_letter(0x00E9));
    CHECK(!is_letter(0x1F600));
    CHECK(!is_digit(0x1F600));
    CHECK(is_letter(0x10400));
    CHECK(is_whitespace(0x2003));
    CHECK(!is_whitespace(0x200B));
    CHECK(gpt2_pre_tokenize(u8"é中𐐀") == std::vector<std::string>({u8"é中𐐀"}));
    CHECK(gpt2_pre_tokenize(u8"١٢٣٤") ==
          std::vector<std::string>({u8"١", u8"٢", u8"٣", u8"٤"}));
    CHECK(gpt2_pre_tokenize(u8"😀é") == std::vector<std::string>({u8"😀é"}));
    CHECK(gpt2_pre_tokenize(u8"\u2003中") == std::vector<std::string>({u8"\u2003中"}));
}

void test_malformed_utf8() {
    const std::string truncated_four_byte("\xF0", 1);
    const std::string truncated_sequence("\xF0\x9F", 2);
    CHECK(gpt2_pre_tokenize(truncated_four_byte) ==
          std::vector<std::string>({truncated_four_byte}));
    int advance = 0;
    CHECK(utf8_codepoint(truncated_sequence.data(),
                         static_cast<int>(truncated_sequence.size()), &advance) == 0xF0);
    CHECK(advance == 1);
}

void test_model_pair_resolution() {
    namespace fs = std::filesystem;
    using tts_cpp::minimax::detail::ModelPair;
    using tts_cpp::minimax::detail::resolve_model_pair;
    const fs::path root = fs::path("/tmp/tether") /
                          ("minimax-model-pair-" + std::to_string(std::random_device{}()));
    fs::create_directories(root / "mm3");
    touch(root / "mm3-lm-f16.gguf");
    touch(root / "mm3-synth-f16.gguf");
    touch(root / "MM3-LM-Q8_0.GGUF");
    touch(root / "mm3-SYNTH-q8_0.gguf");
    ModelPair pair = resolve_model_pair(root.string(), "", "");
    CHECK(pair.quant == "q8_0");
    CHECK(fs::path(pair.lm).filename() == "MM3-LM-Q8_0.GGUF");
    touch(root / "mm3" / "mm3-lm-q8_0.gguf");
    bool ambiguous = false;
    try {
        resolve_model_pair(root.string(), "", "");
    } catch (const std::runtime_error &) {
        ambiguous = true;
    }
    CHECK(ambiguous);
    fs::remove_all(root);
}

void test_backend_configuration() {
    using tts_cpp::minimax::detail::backend_configuration_matches;
    CHECK(backend_configuration_matches(0, 4, "first", 8, "second"));
    CHECK(backend_configuration_matches(1, 4, "backends/.", 4, "backends"));
    CHECK(!backend_configuration_matches(1, 4, "backends", 8, "backends"));
    CHECK(!backend_configuration_matches(1, 4, "first", 4, "second"));
}

void test_engine_instance_limit() {
    using tts_cpp::minimax::detail::engine_instance_available;
    CHECK(engine_instance_available(0));
    CHECK(!engine_instance_available(1));
}

void test_cancellation() {
    using tts_cpp::minimax::detail::cancellation_requested;
    CHECK(!cancellation_requested({}));
    CHECK(cancellation_requested([] { return true; }));
    CHECK(!cancellation_requested([] { return false; }));
}

void test_progress_cancellation() {
    bool cancelled = false;
    std::string error;
    const MM3ProgressCb progress = [&cancelled](const MM3GenProgress &) {
        cancelled = true;
    };
    CHECK(!mm3_emit_progress(progress, {"stitch", -1, 1, 0, 1},
                             [&cancelled] { return cancelled; }, &error));
    CHECK(error == MM3_ERR_CANCELLED);
}

}

int main() {
    test_frame_validation();
    test_prompt();
    test_unconditional_mask();
    test_noise();
    test_flow_schedule();
    test_condition_length();
    test_window_arithmetic();
    test_sampling_edges();
    test_model_compatibility();
    test_unicode_categories();
    test_malformed_utf8();
    test_model_pair_resolution();
    test_backend_configuration();
    test_engine_instance_limit();
    test_cancellation();
    test_progress_cancellation();
    std::fprintf(stderr, "[test-minimax-units] %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
