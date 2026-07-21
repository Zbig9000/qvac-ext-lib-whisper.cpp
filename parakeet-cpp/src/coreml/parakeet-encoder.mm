// Core ML FastConformer encoder sidecar for parakeet-cpp.
//
// Derived from the whisper.cpp Core ML wrapper pattern (src/coreml/whisper-encoder.mm),
// generalised to load an arbitrary compiled encoder via the generic MLModel API so it
// works with whatever `.mlmodelc` the mobius export produces (no code-generated model
// interface required). Tensor orientation is discovered from the model description, so
// the export only has to agree on dimension sizes, not their order.

#if !__has_feature(objc_arc)
#error "parakeet-encoder.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-encoder.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace {

float decode_float16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int      shift = 0;
            uint32_t m     = mant;
            do { m <<= 1; ++shift; } while ((m & 0x400u) == 0);
            bits = sign | ((uint32_t) (127 - 15 - shift + 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

NSString * sole_multiarray_feature(NSDictionary<NSString *, MLFeatureDescription *> * descs) {
    NSString * found = nil;
    for (NSString * key in descs) {
        if (descs[key].type == MLFeatureTypeMultiArray) {
            if (found != nil) return nil;
            found = key;
        }
    }
    return found;
}

bool shape_is_concrete(NSArray<NSNumber *> * shape) {
    if (shape.count < 2) return false;
    for (NSNumber * dim in shape) {
        if (dim.longLongValue <= 0) return false;
    }
    return true;
}

// Matches the last two dims of `shape` against the unordered pair {rows, cols}.
// Sets `transpose` to false for a [.., rows, cols] layout and true for [.., cols, rows].
bool match_trailing_dims(NSArray<NSNumber *> * shape, int64_t rows, int64_t cols, bool * transpose) {
    const NSUInteger n = shape.count;
    if (n < 2) return false;
    const int64_t d_outer = shape[n - 2].longLongValue;
    const int64_t d_inner = shape[n - 1].longLongValue;
    if (d_outer == rows && d_inner == cols) { *transpose = false; return true; }
    if (d_outer == cols && d_inner == rows) { *transpose = true;  return true; }
    return false;
}

// Resolves the concrete input shape to allocate and whether it is feature-major.
// Fixed model shapes are honoured as-is; flexible shapes default to the NeMo-natural
// features-major [1, n_mels, n_mel_frames]. Returns nil on a fixed-shape size mismatch.
NSArray<NSNumber *> * resolve_input_shape(NSArray<NSNumber *> * declared,
                                          int64_t n_mel_frames, int64_t n_mels,
                                          bool * transpose) {
    if (declared != nil && shape_is_concrete(declared)) {
        if (!match_trailing_dims(declared, n_mel_frames, n_mels, transpose)) return nil;
        return declared;
    }
    *transpose = true;
    return @[ @1, @(n_mels), @(n_mel_frames) ];
}

MLMultiArray * build_input_array(NSArray<NSNumber *> * shape, bool transpose,
                                 const float * mel, int64_t n_mel_frames, int64_t n_mels,
                                 NSError ** err) {
    MLMultiArray * arr = [[MLMultiArray alloc] initWithShape:shape
                                                    dataType:MLMultiArrayDataTypeFloat32
                                                       error:err];
    if (arr == nil) return nil;

    float * dst = (float *) arr.dataPointer;
    if (!transpose) {
        std::memcpy(dst, mel, (size_t) n_mel_frames * n_mels * sizeof(float));
        return arr;
    }
    for (int64_t m = 0; m < n_mels; ++m) {
        for (int64_t t = 0; t < n_mel_frames; ++t) {
            dst[m * n_mel_frames + t] = mel[t * n_mels + m];
        }
    }
    return arr;
}

bool read_scalar(const void * base, MLMultiArrayDataType dt, int64_t off, float * out) {
    switch (dt) {
        case MLMultiArrayDataTypeFloat32: *out = ((const float *)    base)[off];                 return true;
        case MLMultiArrayDataTypeDouble:  *out = (float) ((const double *) base)[off];            return true;
        case MLMultiArrayDataTypeFloat16: *out = decode_float16(((const uint16_t *) base)[off]);  return true;
        default:                          return false;
    }
}

// Copies the model output into `dst` as row-major (n_enc_frames, d_model), adapting to the
// output tensor orientation and dtype and honouring its element strides.
bool copy_output_array(MLMultiArray * arr, float * dst, int64_t n_enc_frames, int64_t d_model) {
    if (arr == nil) return false;

    const void               * base  = arr.dataPointer;
    const MLMultiArrayDataType dt    = arr.dataType;
    const int64_t              total = n_enc_frames * d_model;

    bool transpose = false;
    if (match_trailing_dims(arr.shape, n_enc_frames, d_model, &transpose)) {
        const NSUInteger      n       = arr.shape.count;
        NSArray<NSNumber *> * strides = arr.strides;
        const int64_t         s_outer = strides[n - 2].longLongValue;
        const int64_t         s_inner = strides[n - 1].longLongValue;
        for (int64_t t = 0; t < n_enc_frames; ++t) {
            for (int64_t f = 0; f < d_model; ++f) {
                const int64_t outer = transpose ? f : t;
                const int64_t inner = transpose ? t : f;
                float value;
                if (!read_scalar(base, dt, outer * s_outer + inner * s_inner, &value)) return false;
                dst[t * d_model + f] = value;
            }
        }
        return true;
    }

    // Unmatched shape but matching element count: assume a contiguous row-major
    // (n_enc_frames, d_model) buffer -- the natural layout for a flattened export.
    if ((int64_t) arr.count != total) return false;
    for (int64_t i = 0; i < total; ++i) {
        float value;
        if (!read_scalar(base, dt, i, &value)) return false;
        dst[i] = value;
    }
    return true;
}

}  // namespace

struct parakeet_coreml_context {
    const void * model = nullptr;  // CFBridgingRetain'd MLModel *
    std::string  input_name;
    std::string  output_name;
    std::string  label;
};

struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc) {
    if (path_mlmodelc == nullptr) return nullptr;

    @autoreleasepool {
        NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
        if (path == nil) return nullptr;

        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        NSError * err   = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                            configuration:config
                                                    error:&err];
        if (model == nil || err != nil) return nullptr;

        NSString * in_name  = sole_multiarray_feature(model.modelDescription.inputDescriptionsByName);
        NSString * out_name = sole_multiarray_feature(model.modelDescription.outputDescriptionsByName);
        if (in_name == nil || out_name == nil) return nullptr;

        auto * ctx = new parakeet_coreml_context();
        ctx->model       = CFBridgingRetain(model);
        ctx->input_name  = in_name.UTF8String;
        ctx->output_name = out_name.UTF8String;
        ctx->label       = "coreml";
        return ctx;
    }
}

void parakeet_coreml_free(struct parakeet_coreml_context * ctx) {
    if (ctx == nullptr) return;
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out) {
    if (ctx == nullptr || ctx->model == nullptr || mel == nullptr || encoder_out == nullptr) return 1;
    if (n_mel_frames <= 0 || n_mels <= 0 || n_enc_frames <= 0 || d_model <= 0) return 1;

    @autoreleasepool {
        MLModel  * model    = (__bridge MLModel *) ctx->model;
        NSString * in_name  = [NSString stringWithUTF8String:ctx->input_name.c_str()];
        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];

        MLMultiArrayConstraint * in_constraint =
            model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;

        bool                  in_transpose = true;
        NSArray<NSNumber *> * in_shape     =
            resolve_input_shape(in_constraint.shape, n_mel_frames, n_mels, &in_transpose);
        if (in_shape == nil) return 2;

        NSError      * err    = nil;
        MLMultiArray * in_arr = build_input_array(in_shape, in_transpose, mel, n_mel_frames, n_mels, &err);
        if (in_arr == nil || err != nil) return 3;

        MLDictionaryFeatureProvider * provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ in_name : in_arr } error:&err];
        if (provider == nil || err != nil) return 4;

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&err];
        if (result == nil || err != nil) return 5;

        MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
        if (out_arr == nil) return 6;
        if (!copy_output_array(out_arr, encoder_out, n_enc_frames, d_model)) return 7;
        return 0;
    }
}

const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}
