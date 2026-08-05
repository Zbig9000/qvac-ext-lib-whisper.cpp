// Audio8 end-to-end parity through the public engine API.
//
// The stage tests pin the language model and the codec separately against
// their own fixtures. This one drives the same path a caller drives -- text in,
// waveform out -- so it also covers what only shows up when the stages are
// joined: the ChatML prompt the tokenizer builds, the frame budget, the
// codebook-major transposition between the two halves, and, on the cloning
// side, the engine encoding a WAV into the speaker history by itself.
//
// Greedy decoding throughout, because that is the only trajectory the fixtures
// can pin; the sampler has its own test.

#include "tts-cpp/audio8/engine.h"

#include "json.hpp"
#include "npy.h"
#include "voice_features.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

// The waveform is bounded by a tanh and the codec test already holds it to a
// few times 1e-6; the extra room is for the language model's own reassociation
// reaching the codec through identical codes.
constexpr double WAVEFORM_TOLERANCE = 5e-5;

struct fixture {
    std::string dir;

    npy_array load(const std::string & name) const {
        return npy_load(dir + "/" + name + ".npy");
    }

    nlohmann::json meta() const {
        std::ifstream file(dir + "/meta.json");
        if (!file) throw std::runtime_error("cannot open " + dir + "/meta.json");
        return nlohmann::json::parse(file);
    }
};

struct paths {
    std::string lm;
    std::string decoder;
    std::string encoder;
    std::string reference_wav;
    int threads = 4;
};

const float * as_f32(const npy_array & array) {
    return reinterpret_cast<const float *>(array.data.data());
}

tts_cpp::audio8::EngineOptions options_for(const paths & where, const nlohmann::json & meta) {
    tts_cpp::audio8::EngineOptions opts;
    opts.lm_gguf_path = where.lm;
    opts.codec_decoder_gguf_path = where.decoder;
    opts.codec_encoder_gguf_path = where.encoder;
    opts.n_threads = where.threads;
    opts.greedy = true;
    opts.max_frames = meta.at("max_new_tokens").get<int>();
    return opts;
}

tts_cpp::audio8::VoicePrompt voice_from(const std::string & wav_path,
                                        const nlohmann::json & meta) {
    tts_cpp::audio8::VoicePrompt voice;
    if (!wav_load(wav_path, voice.pcm, voice.sample_rate)) {
        throw std::runtime_error("cannot read " + wav_path);
    }
    voice.transcript = meta.at("reference_text").get<std::string>();
    return voice;
}

bool check_waveform(const char * tag, const std::vector<float> & got,
                    const npy_array & want) {
    if (got.size() != want.n_elements()) {
        std::fprintf(stderr, "%s: FAIL %zu samples, reference has %zu\n", tag, got.size(),
                     want.n_elements());
        return false;
    }
    const compare_stats stats = compare_f32(got.data(), as_f32(want), got.size());
    print_compare(tag, stats);
    if (!std::isfinite(stats.max_abs_err)) {
        std::fprintf(stderr, "%s: FAIL non-finite samples\n", tag);
        return false;
    }
    if (stats.max_abs_err > WAVEFORM_TOLERANCE) {
        std::fprintf(stderr, "%s: FAIL max|delta| %.3e > %.1e\n", tag, stats.max_abs_err,
                     WAVEFORM_TOLERANCE);
        return false;
    }
    return true;
}

bool check_frames(const char * tag, int got, const nlohmann::json & meta) {
    const int want = meta.at("generated_frames").get<int>();
    if (got == want) return true;
    std::fprintf(stderr, "%s: FAIL %d frames, reference emitted %d\n", tag, got, want);
    return false;
}

bool run_case(const char * tag, const paths & where, const fixture & data, bool cloning) {
    const nlohmann::json meta = data.meta();
    tts_cpp::audio8::Engine engine(options_for(where, meta));
    const std::string text = meta.at("text").get<std::string>();

    const tts_cpp::audio8::SynthesisResult result =
        cloning ? engine.synthesize(text, voice_from(where.reference_wav, meta))
                : engine.synthesize(text);
    std::printf("%s: %d frames, %.2f s at %d Hz\n", tag, result.frames, result.duration_s,
                result.sample_rate);

    bool ok = check_frames(tag, result.frames, meta);
    ok &= check_waveform(tag, result.pcm, data.load("wav"));
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s <lm.gguf> <codec-decoder.gguf> <codec-encoder.gguf> "
                     "<ref-dir> <clone-ref-dir> <reference.wav> [threads]\n",
                     argv[0]);
        return 1;
    }
    paths where;
    where.lm = argv[1];
    where.decoder = argv[2];
    where.encoder = argv[3];
    where.reference_wav = argv[6];
    where.threads = argc > 7 ? std::atoi(argv[7]) : 4;

    try {
        bool ok = run_case("text", where, fixture{argv[4]}, /*cloning=*/false);
        ok &= run_case("clone", where, fixture{argv[5]}, /*cloning=*/true);
        std::printf("\n%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & failure) {
        std::fprintf(stderr, "engine: %s\n", failure.what());
        return 1;
    }
}
