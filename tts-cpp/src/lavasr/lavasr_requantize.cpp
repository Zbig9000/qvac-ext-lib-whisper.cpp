// lavasr-requantize: block-quantize the big 2-D matmul weights of a LavaSR
// enhancer GGUF to a smaller on-disk type, copying everything else through
// byte-for-byte.
//
// This is the C++ companion to scripts/requantize-gguf.py.  The Python gguf
// library can only quantize the legacy tiers (q4_0/q5_0/q8_0); the K-quants
// (q4_K/q5_K/q6_K) — which give a better accuracy/size trade-off — are only
// reachable through ggml's own quantizer (ggml_quantize_chunk).  The enhancer
// loader (enhancer_gguf.cpp) already reads any quantized type generically via
// ggml_get_type_traits()->to_float ("dequant-at-load"), so producing the file
// is the only missing half — which is what this tool is for.  (QVAC-21906.)
//
// Usage:
//   lavasr-requantize <in.gguf> <out.gguf> <q4_0|q5_0|q8_0|q4_K|q5_K|q6_K>
//
// The source should be the F16 (or F32) enhancer GGUF from
// convert-lavasr-enhancer-to-gguf.py.  Selection mirrors requantize-gguf.py's
// should_quantize() 2-D path: a tensor is quantized iff it is a 2-D float tensor
// (ne[2]==ne[3]==1), has >= 1024 elements, and its reduction dim ne0 is a
// multiple of the target's block size.  For the enhancer that is exactly the
// 8x ConvNeXt pwconv1/pwconv2 and the spec-head Linear (17 tensors); the K=7
// conv kernels stay F16 and the LayerNorm scales / biases / layer-scale gamma
// stay F32.  All KV metadata is copied verbatim (gguf_set_kv).

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct TypeName {
    const char * name;
    ggml_type    type;
};

// The block-quant tiers this tool can emit.  Legacy tiers overlap with
// requantize-gguf.py so the two stay interchangeable; the K-quants are the
// value-add.
constexpr TypeName kTypes[] = {
    {"q4_0", GGML_TYPE_Q4_0}, {"q5_0", GGML_TYPE_Q5_0}, {"q8_0", GGML_TYPE_Q8_0},
    {"q4_K", GGML_TYPE_Q4_K}, {"q5_K", GGML_TYPE_Q5_K}, {"q6_K", GGML_TYPE_Q6_K},
};

bool parse_type(const std::string & s, ggml_type & out) {
    for (const TypeName & t : kTypes) {
        if (s == t.name) {
            out = t.type;
            return true;
        }
    }
    return false;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s <in.gguf> <out.gguf> "
                 "<q4_0|q5_0|q8_0|q4_K|q5_K|q6_K>\n",
                 argv0);
}

// Dequantize a source tensor's payload to F32.  Only F32/F16 sources are
// selected for quantization, but route F16 through the public type-traits
// entry point so the tool never hard-codes a half->float loop.
bool to_f32(const ggml_tensor * src, std::vector<float> & out) {
    const int64_t n = ggml_nelements(src);
    out.resize(static_cast<size_t>(n));
    if (src->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), src->data, static_cast<size_t>(n) * sizeof(float));
        return true;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(src->type);
    if (!tr || !tr->to_float) {
        return false;
    }
    tr->to_float(src->data, out.data(), n);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        usage(argv[0]);
        return 2;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    ggml_type         qtype    = GGML_TYPE_F32;
    if (!parse_type(argv[3], qtype)) {
        std::fprintf(stderr, "error: unknown target type '%s'\n", argv[3]);
        usage(argv[0]);
        return 2;
    }

    ggml_context *   in_ctx = nullptr;
    gguf_init_params ip     = {/*no_alloc=*/false, &in_ctx};
    gguf_context *   in     = gguf_init_from_file(in_path.c_str(), ip);
    if (!in || !in_ctx) {
        std::fprintf(stderr, "error: failed to read GGUF '%s'\n", in_path.c_str());
        return 1;
    }

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);  // copy every KV pair verbatim

    // Output tensor arena: the source total is a safe upper bound because
    // quantized tensors only ever shrink and everything else is copied as-is.
    const int64_t n_tensors = gguf_get_n_tensors(in);
    size_t        arena     = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        arena += ggml_nbytes(ggml_get_tensor(in_ctx, gguf_get_tensor_name(in, i)));
    }
    ggml_init_params op = {arena + static_cast<size_t>(n_tensors + 1) * ggml_tensor_overhead() +
                               (1u << 20),
                           nullptr, /*no_alloc=*/false};
    ggml_context * out_ctx = ggml_init(op);
    if (!out_ctx) {
        std::fprintf(stderr, "error: ggml_init for output failed\n");
        gguf_free(out);
        gguf_free(in);
        return 1;
    }

    const int64_t block   = ggml_blck_size(qtype);
    int           n_quant = 0, n_copy = 0;
    int           rc      = 0;

    std::printf("requantize %s -> %s (target %s, block=%" PRId64 ")\n", in_path.c_str(),
                out_path.c_str(), argv[3], block);

    for (int64_t i = 0; i < n_tensors && rc == 0; i++) {
        const char *  name = gguf_get_tensor_name(in, i);
        ggml_tensor * src  = ggml_get_tensor(in_ctx, name);
        const int     ndim = ggml_n_dims(src);

        const bool is_2d    = src->ne[2] == 1 && src->ne[3] == 1 && src->ne[1] > 1 && src->ne[0] > 1;
        const bool is_float = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16;
        const bool aligned  = (src->ne[0] % block) == 0;
        const bool big      = ggml_nelements(src) >= 1024;

        if (is_2d && is_float && aligned && big) {
            std::vector<float> f32;
            if (!to_f32(src, f32)) {
                std::fprintf(stderr, "error: cannot dequantize source tensor '%s'\n", name);
                rc = 1;
                break;
            }
            ggml_tensor * dst = ggml_new_tensor(out_ctx, qtype, ndim, src->ne);
            ggml_set_name(dst, name);
            const int64_t n_per_row = src->ne[0];
            const int64_t nrows     = ggml_nelements(src) / n_per_row;
            const size_t  wrote =
                ggml_quantize_chunk(qtype, f32.data(), dst->data, 0, nrows, n_per_row, nullptr);
            if (wrote != ggml_nbytes(dst)) {
                std::fprintf(stderr, "error: quantize '%s' wrote %zu bytes, expected %zu\n", name,
                             wrote, ggml_nbytes(dst));
                rc = 1;
                break;
            }
            gguf_add_tensor(out, dst);
            n_quant++;
            std::printf("  quant %-40s %-4s -> %-4s  ne=[%" PRId64 ",%" PRId64 "]\n", name,
                        ggml_type_name(src->type), ggml_type_name(qtype), src->ne[0], src->ne[1]);
        } else {
            ggml_tensor * dst = ggml_new_tensor(out_ctx, src->type, ndim, src->ne);
            ggml_set_name(dst, name);
            std::memcpy(dst->data, src->data, ggml_nbytes(src));
            gguf_add_tensor(out, dst);
            n_copy++;
        }
    }

    if (rc == 0 && !gguf_write_to_file(out, out_path.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "error: failed to write '%s'\n", out_path.c_str());
        rc = 1;
    }

    ggml_free(out_ctx);
    gguf_free(out);
    gguf_free(in);

    if (rc == 0) {
        std::printf("OK: %d tensor(s) quantized, %d copied through -> %s\n", n_quant, n_copy,
                    out_path.c_str());
    }
    return rc;
}
