#include "tts-cpp/supertonic/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model supertonic2.gguf --text TEXT --out out.wav\n"
        "          [--language en] [--voice NAME] [--steps N] [--speed X]\n"
        "          (voice/steps/speed default to GGUF metadata when omitted)\n"
        "          [--seed 42] [--threads N] [--n-gpu-layers N]\n"
        "          [--vulkan-device N] (Vulkan adapter index; ignored unless\n"
        "                            built with -DGGML_VULKAN=ON; default 0,\n"
        "                            -1 = auto-pick adapter with most free VRAM)\n"
        "          [--f16-attn 0|1] (vector-estimator F16 K/V attention;\n"
        "                            defaults to auto: on for GPU, off for CPU)\n"
        "          [--kv-attn-type auto|f32|f16|bf16|q8_0]\n"
        "                            (vector-estimator multi-dtype K/V flash-attn;\n"
        "                            generalises --f16-attn.  default auto: falls\n"
        "                            back to --f16-attn for backwards-compat.\n"
        "                            bf16/q8_0 require Vulkan adapter support;\n"
        "                            silent fallback to f32 on probe miss.)\n"
        "          [--f16-weights 0|1] (load-time F16 materialization for the\n"
        "                            audit-identified hot matmul / pwconv weights;\n"
        "                            defaults to auto: on for GPU, off for CPU)\n"
        "          [--f16-weights-deny PATTERN1,PATTERN2,...] (substring patterns,\n"
        "                            comma-separated; matching tensors stay F32 even\n"
        "                            when --f16-weights is on.  Default empty.)\n"
        "          [--prewarm TEXT] (run one throwaway synth on TEXT at engine\n"
        "                            construction so first-real-call latency on\n"
        "                            Vulkan / OpenCL doesn't pay the shader-\n"
        "                            compile cost; no-op on CPU)\n"
        "          [--vulkan-prefer-host-memory]    (sets GGML_VK_PREFER_HOST_MEMORY=1)\n"
        "          [--vulkan-disable-coopmat2]      (sets GGML_VK_DISABLE_COOPMAT2=1)\n"
        "          [--vulkan-disable-bfloat16]      (sets GGML_VK_DISABLE_BFLOAT16=1)\n"
        "          [--vulkan-perf-logger]           (sets GGML_VK_PERF_LOGGER=1)\n"
        "          [--vulkan-async-transfer]        (sets GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1)\n"
        "          [--vulkan-env KEY=VALUE]         (set arbitrary GGML_VK_* env var;\n"
        "                            may be repeated; operator-set env vars in the shell\n"
        "                            STILL win over these CLI overrides)\n"
        "          [--noise-npy /path/to/noise.npy]\n",
        argv0);
}

void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output wav: " + path);
    uint32_t n = (uint32_t) wav.size();
    uint32_t byte_rate = (uint32_t) sr * 2;
    uint32_t data_size = n * 2;
    uint32_t chunk_size = 36 + data_size;
    uint16_t fmt = 1, channels = 1, align = 2, bps = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f); std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, f); std::fwrite(&fmt, 2, 1, f);
    std::fwrite(&channels, 2, 1, f); std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f); std::fwrite(&align, 2, 1, f);
    std::fwrite(&bps, 2, 1, f); std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float c = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t) std::lrintf(c * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

} // namespace

int main(int argc, char ** argv) {
    tts_cpp::supertonic::EngineOptions opts;
    std::string text;
    std::string out;
    // QVAC-18605 round 4 — wrap arg parse in try/catch so invalid
    // values (`--kv-attn-type bogus`, `--vulkan-device abc`, etc.)
    // surface as a clean `error: ...` line + exit 2 instead of an
    // uncaught-exception backtrace.  Same exit-code convention as
    // unknown-flag / missing-required handling below.
    try {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
            return argv[++i];
        };
        if (arg == "--model") opts.model_gguf_path = next("--model");
        else if (arg == "--text") text = next("--text");
        else if (arg == "--out") out = next("--out");
        else if (arg == "--language") opts.language = next("--language");
        else if (arg == "--voice") opts.voice = next("--voice");
        else if (arg == "--steps") opts.steps = std::stoi(next("--steps"));
        else if (arg == "--speed") opts.speed = std::stof(next("--speed"));
        else if (arg == "--seed") opts.seed = std::stoi(next("--seed"));
        else if (arg == "--threads") opts.n_threads = std::stoi(next("--threads"));
        else if (arg == "--n-gpu-layers") opts.n_gpu_layers = std::stoi(next("--n-gpu-layers"));
        else if (arg == "--vulkan-device") opts.vulkan_device = std::stoi(next("--vulkan-device"));
        else if (arg == "--f16-attn") opts.f16_attn = std::stoi(next("--f16-attn"));
        else if (arg == "--kv-attn-type") {
            const std::string v = next("--kv-attn-type");
            if      (v == "auto") opts.kv_attn_type = -1;
            else if (v == "f32")  opts.kv_attn_type = 0;
            else if (v == "f16")  opts.kv_attn_type = 1;
            else if (v == "bf16") opts.kv_attn_type = 2;
            else if (v == "q8_0") opts.kv_attn_type = 3;
            else throw std::runtime_error(
                "--kv-attn-type expects one of: auto, f32, f16, bf16, q8_0 (got: " + v + ")");
        }
        else if (arg == "--f16-weights") opts.f16_weights = std::stoi(next("--f16-weights"));
        else if (arg == "--f16-weights-deny") {
            // Comma-split into a vector<string>.  Empty entries
            // are tolerated (predicate skips them defensively).
            opts.f16_weights_deny_list.clear();
            const std::string raw = next("--f16-weights-deny");
            size_t start = 0;
            for (size_t k = 0; k <= raw.size(); ++k) {
                if (k == raw.size() || raw[k] == ',') {
                    opts.f16_weights_deny_list.emplace_back(raw.substr(start, k - start));
                    start = k + 1;
                }
            }
        }
        else if (arg == "--prewarm") opts.prewarm_text = next("--prewarm");
        else if (arg == "--vulkan-prefer-host-memory") opts.vulkan_env_overrides["GGML_VK_PREFER_HOST_MEMORY"]      = "1";
        else if (arg == "--vulkan-disable-coopmat2")   opts.vulkan_env_overrides["GGML_VK_DISABLE_COOPMAT2"]        = "1";
        else if (arg == "--vulkan-disable-bfloat16")   opts.vulkan_env_overrides["GGML_VK_DISABLE_BFLOAT16"]        = "1";
        else if (arg == "--vulkan-perf-logger")        opts.vulkan_env_overrides["GGML_VK_PERF_LOGGER"]             = "1";
        else if (arg == "--vulkan-async-transfer")     opts.vulkan_env_overrides["GGML_VK_ASYNC_USE_TRANSFER_QUEUE"]= "1";
        else if (arg == "--vulkan-env") {
            const std::string raw = next("--vulkan-env");
            const auto eq = raw.find('=');
            if (eq == std::string::npos || eq == 0) {
                throw std::runtime_error("--vulkan-env expects KEY=VALUE (got: " + raw + ")");
            }
            opts.vulkan_env_overrides[raw.substr(0, eq)] = raw.substr(eq + 1);
        }
        else if (arg == "--noise-npy") opts.noise_npy_path = next("--noise-npy");
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", arg.c_str()); usage(argv[0]); return 2; }
    }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0]);
        return 2;
    }
    if (opts.model_gguf_path.empty() || text.empty() || out.empty()) {
        usage(argv[0]);
        return 2;
    }
    try {
        auto result = tts_cpp::supertonic::synthesize(opts, text);
        write_wav(out, result.pcm, result.sample_rate);
        fprintf(stderr, "wrote %s (%.2fs @ %d Hz, %zu samples)\n",
                out.c_str(), result.duration_s, result.sample_rate, result.pcm.size());
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
