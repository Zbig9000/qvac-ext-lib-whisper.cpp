#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace tts_cpp::minimax::detail {

constexpr int kDefaultFrameRate = 25;
constexpr int kDefaultMaxFrames = 9000;
constexpr int kDefaultFlowSteps = 30;
constexpr float kDefaultCfgScale = 1.7f;
constexpr int kWindowFrames = 200;
constexpr int kHopFrames = 100;
constexpr int kCarryLatents = 344;
constexpr int kBlendLatents = 172;
constexpr int kCropLeftLatents = 86;
constexpr int kCropRightLatents = 258;

struct ConditionRate {
    int input_sampling_rate = 24000;
    int input_hop_length = 960;
    int output_sampling_rate = 44100;
    int output_hop_length = 512;
};

struct CropSpan {
    int64_t left = 0;
    int64_t length = 0;
};

struct ModelCompatibility {
    int64_t lm_embedding = 0;
    int64_t lm_codebooks = 0;
    int64_t lm_acoustic_vocab = 0;
    int64_t frame_rate = 0;
    int64_t max_audio_frames = 0;
    int64_t max_prompt_tokens = 0;
    int64_t depth_embedding = 0;
    int64_t depth_codebooks = 0;
    int64_t depth_acoustic_vocab = 0;
    int64_t condition_layers = 0;
    int64_t condition_hidden = 0;
    int64_t condition_out = 0;
    ConditionRate condition_rate{0, 0, 0, 0};
    int64_t dit_condition = 0;
    int64_t dit_channels = 0;
    int64_t window_frames = 0;
    int64_t hop_frames = 0;
    int64_t window_latents = 0;
    int64_t hop_latents = 0;
    int64_t vocoder_latent_channels = 0;
    int64_t vocoder_sampling_rate = 0;
    int64_t vocoder_channels = 0;
    int64_t vocoder_upsample = 0;
    std::vector<std::string> components;
};

struct ModelPair {
    std::string lm;
    std::string synth;
    std::string quant;
};

int64_t frames_from_duration(double duration, int frame_rate, int max_frames);
int64_t validate_frames(int64_t frames, int max_frames);
std::string build_prompt(const std::string & caption, const std::string & lyrics);
std::vector<int32_t> mask_unconditional(const std::vector<int32_t> & conditional, int32_t mask_token);
void fill_noise(uint64_t seed, int64_t window, std::vector<float> & output, int64_t count);
void flow_schedule(int steps, std::vector<float> & sigmas, std::vector<float> & timesteps);
int64_t condition_latent_length(const ConditionRate & rate, int64_t frames);
std::vector<int64_t> window_starts(int64_t frames, int64_t window_frames, int64_t hop_frames);
CropSpan crop_span(int64_t latent_length, int64_t window_index, int64_t window_count, int64_t upsample);
int64_t stitched_sample_count(const std::vector<int64_t> & latent_lengths, int64_t upsample);
int64_t sample_top_k(const float * logits, int64_t count, int top_k, std::mt19937_64 & random);
std::vector<std::string> validate_model_compatibility(const ModelCompatibility & model);
ModelPair resolve_model_pair(const std::string & model_dir, const std::string & explicit_lm,
                             const std::string & explicit_synth);
bool backend_configuration_matches(int active_references, int active_threads,
                                   const std::string & active_backends_dir, int requested_threads,
                                   const std::string & requested_backends_dir);
bool engine_instance_available(int active_instances);
bool cancellation_requested(const std::function<bool()> & callback);

}
