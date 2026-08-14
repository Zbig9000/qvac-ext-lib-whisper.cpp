#include "minimax/logic.h"

#include "minimax/mm3-sample.h"
#include "minimax/request-utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <map>
#include <stdexcept>

namespace tts_cpp::minimax::detail {
namespace {

namespace fs = std::filesystem;

uint64_t splitmix64(uint64_t & state) {
    uint64_t value = (state += UINT64_C(0x9E3779B97F4A7C15));
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

void fill_noise_pairs(uint64_t & state, std::vector<float> & output) {
    constexpr double kUnitScale = 1.0 / 9007199254740992.0;
    constexpr double kMinimumUniform = 1e-300;
    constexpr double kTau = 6.283185307179586476925286766559;
    for (size_t index = 0; index < output.size(); index += 2) {
        double first = 0.0;
        do {
            first = static_cast<double>(splitmix64(state) >> 11) * kUnitScale;
        } while (first <= kMinimumUniform);
        const double second = static_cast<double>(splitmix64(state) >> 11) * kUnitScale;
        const double radius = std::sqrt(-2.0 * std::log(first));
        const double angle = kTau * second;
        output[index] = static_cast<float>(radius * std::cos(angle));
        if (index + 1 < output.size()) {
            output[index + 1] = static_cast<float>(radius * std::sin(angle));
        }
    }
}

int64_t sum_crop_lengths(const std::vector<int64_t> & latent_lengths, int64_t upsample) {
    int64_t total = 0;
    for (size_t index = 0; index < latent_lengths.size(); ++index) {
        total += crop_span(latent_lengths[index], static_cast<int64_t>(index),
                           static_cast<int64_t>(latent_lengths.size()), upsample)
                     .length;
    }
    return total;
}

void add_error(bool condition, const std::string & message, std::vector<std::string> & errors) {
    if (!condition) {
        errors.push_back(message);
    }
}

void validate_components(const ModelCompatibility & model, std::vector<std::string> & errors) {
    std::vector<std::string> components = model.components;
    std::sort(components.begin(), components.end());
    add_error(components == std::vector<std::string>({"cond", "depth", "dit", "vocoder"}),
              "synth components must be exactly cond, depth, dit, and vocoder", errors);
}

void validate_dimensions(const ModelCompatibility & model, std::vector<std::string> & errors) {
    add_error(model.lm_embedding > 0 && model.lm_embedding == model.depth_embedding,
              "LM embedding dimension must match depth embedding dimension", errors);
    add_error(model.lm_embedding > 0 && model.lm_embedding == model.condition_hidden,
              "LM embedding dimension must match condition hidden dimension", errors);
    add_error(model.lm_codebooks > 0 && model.lm_codebooks == model.depth_codebooks,
              "LM codebook count must match depth codebook count", errors);
    add_error(model.lm_codebooks > 0 && model.lm_codebooks == model.condition_layers,
              "LM codebook count must match condition layer count", errors);
    add_error(model.lm_acoustic_vocab > 0 && model.lm_acoustic_vocab == model.depth_acoustic_vocab,
              "LM acoustic vocabulary must match depth acoustic vocabulary", errors);
    add_error(model.condition_out > 0 && model.condition_out == model.dit_condition,
              "condition output dimension must match DiT condition dimension", errors);
    add_error(model.dit_channels > 0 && model.dit_channels == model.vocoder_latent_channels,
              "DiT latent channels must match vocoder latent channels", errors);
}

bool has_positive_condition_rate(const ModelCompatibility & model) {
    return model.condition_rate.input_sampling_rate > 0 && model.condition_rate.input_hop_length > 0 &&
           model.condition_rate.output_sampling_rate > 0 && model.condition_rate.output_hop_length > 0;
}

void validate_rates(const ModelCompatibility & model, std::vector<std::string> & errors) {
    add_error(model.frame_rate == kDefaultFrameRate, "LM frame rate must be 25", errors);
    add_error(model.max_audio_frames == kDefaultMaxFrames, "LM maximum audio frames must be 9000", errors);
    add_error(model.max_prompt_tokens > 0, "LM maximum prompt tokens must be positive", errors);
    add_error(has_positive_condition_rate(model), "condition rates and hop lengths must be positive", errors);
    if (has_positive_condition_rate(model)) {
        add_error(model.condition_rate.input_sampling_rate ==
                      model.frame_rate * model.condition_rate.input_hop_length,
                  "condition input rate and hop must match LM frame rate", errors);
        add_error(model.condition_rate.output_sampling_rate == model.vocoder_sampling_rate,
                  "condition output rate must match vocoder sampling rate", errors);
        add_error(model.condition_rate.output_hop_length == model.vocoder_upsample,
                  "condition output hop must match vocoder upsample", errors);
    }
    add_error(model.vocoder_sampling_rate > 0, "vocoder sampling rate must be positive", errors);
    add_error(model.vocoder_channels == 2, "vocoder channels must be 2", errors);
}

void validate_windows(const ModelCompatibility & model, std::vector<std::string> & errors) {
    add_error(model.window_frames == kWindowFrames, "DiT window frames must be 200", errors);
    add_error(model.hop_frames == kHopFrames, "DiT hop frames must be 100", errors);
    if (!has_positive_condition_rate(model) || model.window_frames <= 0 || model.hop_frames <= 0) {
        return;
    }
    add_error(model.window_latents == condition_latent_length(model.condition_rate, model.window_frames),
              "DiT window latents do not match the condition rate", errors);
    add_error(model.hop_latents == condition_latent_length(model.condition_rate, model.hop_frames),
              "DiT hop latents do not match the condition rate", errors);
    add_error(model.hop_latents == kCarryLatents, "DiT hop latents must be 344", errors);
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return value;
}

struct CanonicalName {
    std::string role;
    std::string quant;
};

CanonicalName parse_canonical_name(const fs::path & path) {
    const std::string name = lowercase(path.filename().string());
    constexpr const char * kLmPrefix = "mm3-lm-";
    constexpr const char * kSynthPrefix = "mm3-synth-";
    constexpr const char * kSuffix = ".gguf";
    const std::string prefix = name.rfind(kLmPrefix, 0) == 0
                                   ? kLmPrefix
                                   : (name.rfind(kSynthPrefix, 0) == 0 ? kSynthPrefix : "");
    if (prefix.empty() || name.size() <= prefix.size() + std::char_traits<char>::length(kSuffix) ||
        name.compare(name.size() - std::char_traits<char>::length(kSuffix),
                     std::char_traits<char>::length(kSuffix), kSuffix) != 0) {
        return {};
    }
    return {prefix == kLmPrefix ? "lm" : "synth",
            name.substr(prefix.size(), name.size() - prefix.size() - std::char_traits<char>::length(kSuffix))};
}

using CandidateMap = std::map<std::string, std::vector<std::string>>;

void collect_directory_candidates(const fs::path & directory, CandidateMap & lm, CandidateMap & synth) {
    std::error_code error;
    if (!fs::is_directory(directory, error)) {
        return;
    }
    for (const fs::directory_entry & entry : fs::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) {
            continue;
        }
        const CanonicalName name = parse_canonical_name(entry.path());
        if (name.role == "lm") {
            lm[name.quant].push_back(entry.path().string());
        } else if (name.role == "synth") {
            synth[name.quant].push_back(entry.path().string());
        }
    }
}

void reject_duplicate_candidates(const CandidateMap & candidates, const std::string & role) {
    for (const auto & candidate : candidates) {
        if (candidate.second.size() > 1) {
            throw std::runtime_error("minimax engine: ambiguous " + role + " candidates for quant " +
                                     candidate.first);
        }
    }
}

std::string candidate_for_quant(const CandidateMap & candidates, const std::string & quant) {
    const auto match = candidates.find(quant);
    return match == candidates.end() ? "" : match->second.front();
}

ModelPair select_prioritized_pair(const CandidateMap & lm, const CandidateMap & synth) {
    static const std::vector<std::string> priorities = {"q8_0", "f16", "bf16"};
    for (const std::string & quant : priorities) {
        const std::string lm_path = candidate_for_quant(lm, quant);
        const std::string synth_path = candidate_for_quant(synth, quant);
        if (!lm_path.empty() && !synth_path.empty()) {
            return {lm_path, synth_path, quant};
        }
    }
    throw std::runtime_error("minimax engine: no matched canonical LM and synth GGUF pair found");
}

std::string normalized_directory(const std::string & directory) {
    if (directory.empty()) {
        return "";
    }
    fs::path normalized = fs::path(directory).lexically_normal();
    if (!normalized.has_filename() && normalized != normalized.root_path()) {
        normalized = normalized.parent_path();
    }
    return normalized.string();
}

}

int64_t frames_from_duration(double duration, int frame_rate, int max_frames) {
    if (!std::isfinite(duration) || duration <= 0.0) {
        throw std::invalid_argument("duration must be finite and greater than zero");
    }
    if (frame_rate <= 0) {
        throw std::invalid_argument("frame rate must be greater than zero");
    }
    return validate_frames(static_cast<int64_t>(std::llround(duration * frame_rate)), max_frames);
}

int64_t validate_frames(int64_t frames, int max_frames) {
    if (max_frames <= 0) {
        throw std::invalid_argument("maximum frame count must be greater than zero");
    }
    if (frames <= 0) {
        throw std::invalid_argument("frame count must be greater than zero");
    }
    return frames > max_frames ? max_frames : frames;
}

std::string build_prompt(const std::string & caption, const std::string & lyrics) {
    if (mm3_str_blank(caption)) {
        throw std::invalid_argument("caption must not be empty");
    }
    return mm3_assemble_prompt(caption, lyrics);
}

std::vector<int32_t> mask_unconditional(const std::vector<int32_t> & conditional, int32_t mask_token) {
    std::vector<int32_t> result = conditional;
    if (result.size() < 4) {
        return result;
    }
    for (size_t index = 1; index + 2 < result.size(); ++index) {
        result[index] = mask_token;
    }
    return result;
}

void fill_noise(uint64_t seed, int64_t window, std::vector<float> & output, int64_t count) {
    if (window < 0 || count < 0) {
        throw std::invalid_argument("noise window and count must not be negative");
    }
    uint64_t state =
        seed ^ (UINT64_C(0xA24BAED4963EE407) * static_cast<uint64_t>(window + 1));
    splitmix64(state);
    output.resize(static_cast<size_t>(count));
    fill_noise_pairs(state, output);
}

void flow_schedule(int steps, std::vector<float> & sigmas, std::vector<float> & timesteps) {
    if (steps <= 0) {
        throw std::invalid_argument("flow steps must be greater than zero");
    }
    sigmas.assign(static_cast<size_t>(steps) + 1, 0.0f);
    timesteps.assign(static_cast<size_t>(steps), 0.0f);
    const double start = 1.0;
    const double stop = 1.0 / static_cast<double>(steps);
    const double delta = steps > 1 ? (stop - start) / static_cast<double>(steps - 1) : 0.0;
    for (int index = 0; index < steps; ++index) {
        const double linear = index == steps - 1 ? stop : static_cast<double>(index) * delta + start;
        const float value = 1.0f - static_cast<float>(linear);
        sigmas[static_cast<size_t>(index)] = value;
        timesteps[static_cast<size_t>(index)] = value;
    }
    sigmas[static_cast<size_t>(steps)] = 1.0f;
}

int64_t condition_latent_length(const ConditionRate & rate, int64_t frames) {
    if (frames <= 0 || rate.input_sampling_rate <= 0 || rate.input_hop_length <= 0 ||
        rate.output_sampling_rate <= 0 || rate.output_hop_length <= 0) {
        throw std::invalid_argument("condition rate and frame values must be greater than zero");
    }
    const double value = static_cast<double>(frames) * rate.output_sampling_rate /
                         rate.input_sampling_rate * rate.input_hop_length / rate.output_hop_length;
    const int64_t length = static_cast<int64_t>(value);
    return length < 1 ? 1 : length;
}

std::vector<int64_t> window_starts(int64_t frames, int64_t window_frames, int64_t hop_frames) {
    if (frames <= 0 || window_frames <= 0 || hop_frames <= 0 || hop_frames > window_frames) {
        throw std::invalid_argument("window values are invalid");
    }
    std::vector<int64_t> starts;
    if (frames <= window_frames) {
        starts.push_back(0);
        return starts;
    }
    for (int64_t start = 0; start < frames - hop_frames; start += hop_frames) {
        starts.push_back(start);
    }
    return starts;
}

CropSpan crop_span(int64_t latent_length, int64_t window_index, int64_t window_count, int64_t upsample) {
    if (latent_length < 0 || window_index < 0 || window_count <= 0 || window_index >= window_count ||
        upsample <= 0) {
        throw std::invalid_argument("crop values are invalid");
    }
    const int64_t left = window_index == 0 ? 0 : static_cast<int64_t>(kCropLeftLatents) * upsample;
    const int64_t right =
        window_index == window_count - 1 ? 0 : static_cast<int64_t>(kCropRightLatents) * upsample;
    const int64_t available = latent_length * upsample - left - right;
    return {left, available > 0 ? available : 0};
}

int64_t stitched_sample_count(const std::vector<int64_t> & latent_lengths, int64_t upsample) {
    if (latent_lengths.empty()) {
        return 0;
    }
    return sum_crop_lengths(latent_lengths, upsample);
}

int64_t sample_top_k(const float * logits, int64_t count, int top_k, std::mt19937_64 & random) {
    return mm3_sample_top_k(logits, count, top_k, random);
}

std::vector<std::string> validate_model_compatibility(const ModelCompatibility & model) {
    std::vector<std::string> errors;
    validate_components(model, errors);
    validate_dimensions(model, errors);
    validate_rates(model, errors);
    validate_windows(model, errors);
    return errors;
}

ModelPair resolve_model_pair(const std::string & model_dir, const std::string & explicit_lm,
                             const std::string & explicit_synth) {
    const CanonicalName lm_name = parse_canonical_name(explicit_lm);
    const CanonicalName synth_name = parse_canonical_name(explicit_synth);
    if (!explicit_lm.empty() && !explicit_synth.empty()) {
        if (!lm_name.quant.empty() && !synth_name.quant.empty() && lm_name.quant != synth_name.quant) {
            throw std::runtime_error("minimax engine: explicit canonical GGUF files use different quants");
        }
        return {explicit_lm, explicit_synth, lm_name.quant};
    }
    if (model_dir.empty()) {
        throw std::runtime_error("minimax engine: model directory is required when a GGUF path is missing");
    }
    CandidateMap lm;
    CandidateMap synth;
    collect_directory_candidates(fs::path(model_dir) / "mm3", lm, synth);
    collect_directory_candidates(model_dir, lm, synth);
    reject_duplicate_candidates(lm, "LM");
    reject_duplicate_candidates(synth, "synth");
    if (!explicit_lm.empty() || !explicit_synth.empty()) {
        const CanonicalName explicit_name = explicit_lm.empty() ? synth_name : lm_name;
        if (explicit_name.quant.empty()) {
            throw std::runtime_error(
                "minimax engine: an explicit noncanonical GGUF path requires both model paths");
        }
        const std::string counterpart =
            candidate_for_quant(explicit_lm.empty() ? lm : synth, explicit_name.quant);
        if (counterpart.empty()) {
            throw std::runtime_error("minimax engine: matching canonical GGUF counterpart is missing");
        }
        return explicit_lm.empty() ? ModelPair{counterpart, explicit_synth, explicit_name.quant}
                                   : ModelPair{explicit_lm, counterpart, explicit_name.quant};
    }
    return select_prioritized_pair(lm, synth);
}

bool backend_configuration_matches(int active_references, int active_threads,
                                   const std::string & active_backends_dir, int requested_threads,
                                   const std::string & requested_backends_dir) {
    return active_references <= 0 ||
           (active_threads == requested_threads &&
            normalized_directory(active_backends_dir) == normalized_directory(requested_backends_dir));
}

bool engine_instance_available(int active_instances) {
    return active_instances == 0;
}

bool cancellation_requested(const std::function<bool()> & callback) {
    return callback && callback();
}

}
