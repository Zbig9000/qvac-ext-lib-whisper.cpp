#!/usr/bin/env python3
"""Convert the LavaSR enhancer (Vocos bandwidth-extension) ONNX pair into a
single GGUF for the tts-cpp CPU/GGML enhancer.

The enhancer is two ONNX graphs:
  * enhancer_backbone.onnx   mel[B,80,T] -> hidden[B,T,512]
        embed Conv1d(80->512,k7,pad3) -> LayerNorm
        8x ConvNeXt block:
            dwconv Conv1d(512->512,k7,pad3,group=512)
            LayerNorm(eps=1e-6)
            pwconv1 Linear(512->1536) + erf-GELU
            pwconv2 Linear(1536->512)
            *gamma (layer scale) + residual
        final LayerNorm
  * enhancer_spec_head.onnx  hidden[B,T,512] -> real[B,1025,T], imag[B,1025,T]
        Linear(512->2050) -> transpose -> split(1025,1025)
        mag = clip(exp(split0), max=clip_max);  real = mag*cos(split1);  imag = mag*sin(split1)

Linear (MatMul) weights are stored ONNX-side as [in,out]; we transpose them to
[out,in] (PyTorch convention) so the C++ loader reads ggml ne=[in,out] and runs
ggml_mul_mat(W, x) directly.  Conv weights are stored as ONNX [out,in,k].

Quantization (--ftype q4_0 / q5_0 / q8_0): the block-quant tiers reuse the
shared policy in requantize-gguf.py (should_quantize), so the one-step ONNX->
GGUF converter and the requantize-an-existing-GGUF path can never drift (the
drift between them is exactly what broke S3Gen conversion in QVAC-21203). Only
the big 2-D matmul weights qualify: the 8x ConvNeXt pwconv1/pwconv2 (512<->1536)
and the spec-head Linear (512->2050), i.e. 17 tensors and ~97% of the weights.
The K=7 conv kernels (embed + depthwise) are not block-aligned so they stay F16;
LayerNorm scales, all biases and the per-block layer-scale gamma stay F32. The
C++ loader (enhancer_gguf.cpp) dequantizes every tensor to F32 at load, so the
forward math is identical to F32 — the win is a smaller GGUF (Q4_0 ~= 15% of the
F32 GGUF / ~30% of F16).

Usage:
  python convert-lavasr-enhancer-to-gguf.py \
      --backbone  enhancer_backbone.onnx \
      --spec-head enhancer_spec_head.onnx \
      --out       lavasr-enhancer.gguf \
      --ftype     f32            # or f16 / q8_0 / q5_0 / q4_0
"""
import argparse
import importlib.util
import os
import sys

import gguf
import numpy as np
import onnx
from gguf import GGUFWriter
from onnx import numpy_helper

ARCH = "lavasr-enhancer"

# Map the --ftype quant tiers onto gguf quant types. F32/F16 are handled inline
# by store(); these are the block-quant tiers that route through should_quantize.
_QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}


def _load_should_quantize():
    """Import should_quantize() from the sibling requantize-gguf.py so the
    converter and the requantizer share ONE quant policy (name deny-list +
    block-size gates). The hyphen in the filename blocks a plain import, so
    load it by path."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "requantize-gguf.py")
    spec = importlib.util.spec_from_file_location("requantize_gguf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.should_quantize


should_quantize = _load_should_quantize()

# Mel / STFT params (must match src/lavasr/dsp + the @qvac/tts-onnx enhancer).
N_MELS = 80
DIM = 512
FFN_DIM = 1536
N_BLOCKS = 8
KERNEL = 7
N_FFT = 2048
HOP = 512
WIN = 2048
SPEC_BINS = N_FFT // 2 + 1  # 1025
MEL_REF_SR = 44100          # Slaney mel reference rate (Vocos training)
WORK_SR = 48000             # enhancer operates on 48 kHz audio
LN_EPS = 1e-6


def init_map(graph):
    return {t.name: numpy_helper.to_array(t) for t in graph.initializer}


def node_by_output(graph):
    out = {}
    for n in graph.node:
        for o in n.output:
            out[o] = n
    return out


def find_matmul_weight(graph, inits, by_out, bias_name):
    """Given a `*.bias` initializer name added right after a MatMul, return the
    MatMul's weight array (an initializer)."""
    for n in graph.node:
        if n.op_type == "Add" and bias_name in n.input:
            other = [i for i in n.input if i != bias_name][0]
            mm = by_out.get(other)
            if mm is None or mm.op_type != "MatMul":
                raise RuntimeError(f"expected MatMul feeding Add of {bias_name}")
            for i in mm.input:
                if i in inits:
                    return inits[i]
            raise RuntimeError(f"MatMul for {bias_name} has no initializer input")
    raise RuntimeError(f"no Add node consuming bias {bias_name}")


def store(writer, name, arr, ftype, allow_f16=True):
    arr = np.ascontiguousarray(arr)

    # Block-quant tiers (q4_0 / q5_0 / q8_0): defer the per-tensor keep/quant
    # decision to the shared should_quantize policy so this one-step converter
    # and requantize-gguf.py stay consistent. Only the big 2-D matmul weights
    # (pwconv1/pwconv2/spec_head.out) clear the block-size gate; the K=7 conv
    # kernels, LayerNorm scales, biases and gamma fall through to the F16/F32
    # keep path below (matching requantize-gguf.py's "kept at source dtype").
    if ftype in _QUANT_TYPE:
        qtype = _QUANT_TYPE[ftype]
        if arr.dtype == np.float32 and should_quantize(name, tuple(arr.shape), qtype):
            qdata = gguf.quants.quantize(arr, qtype)
            # raw_shape is the BYTE shape (gguf-0.18+ add_tensor_info treats a
            # uint8 raw tensor's inner dim as bytes/row); raw_dtype=Q* carries
            # the element dims. qdata.shape already encodes it — same call
            # requantize-gguf.py uses.
            writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
            print(f"  {name:42s} {qtype.name:8s} {list(arr.shape)}")
            return
        if allow_f16 and arr.ndim >= 2 and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        elif arr.dtype != np.float32 and arr.dtype != np.float16:
            arr = arr.astype(np.float32)
        writer.add_tensor(name, arr)
        print(f"  {name:42s} {str(arr.dtype):8s} {list(arr.shape)}")
        return

    if ftype == "f16" and allow_f16 and arr.ndim >= 2 and arr.dtype == np.float32:
        arr = arr.astype(np.float16)
    elif arr.dtype != np.float32 and arr.dtype != np.float16:
        arr = arr.astype(np.float32)
    writer.add_tensor(name, arr)
    print(f"  {name:42s} {str(arr.dtype):8s} {list(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--spec-head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ftype", choices=["f32", "f16", "q8_0", "q5_0", "q4_0"],
                    default="f32")
    args = ap.parse_args()

    bb = onnx.load(args.backbone, load_external_data=True).graph
    sh = onnx.load(args.spec_head, load_external_data=True).graph
    bi = init_map(bb)
    si = init_map(sh)
    bb_by_out = node_by_output(bb)

    # The spec head clamps the log-magnitude via Clip(exp(x), None, max) before
    # the cos/sin polar reconstruction (see the graph: Exp -> Clip -> Mul). Read
    # the clamp upper bound straight from that Clip node's `max` input (Clip's
    # 3rd input in opset >= 11) rather than scanning all scalar constants, so a
    # re-export carrying another scalar can't silently change the clamp. Falls
    # back to 1000.0 (with a warning) only if the graph shape ever changes.
    clip_max = 1000.0
    clip_nodes = [n for n in sh.node if n.op_type == "Clip"]
    if (len(clip_nodes) == 1 and len(clip_nodes[0].input) >= 3
            and clip_nodes[0].input[2] in si):
        # si values are already numpy arrays (init_map -> numpy_helper.to_array).
        clip_max = float(si[clip_nodes[0].input[2]].reshape(-1)[0])
    else:
        print(f"WARNING: could not uniquely resolve the spec-head Clip max input "
              f"({len(clip_nodes)} Clip node(s)); using fallback clip_max={clip_max}")

    writer = GGUFWriter(args.out, ARCH)
    writer.add_uint32("lavasr.enhancer.dim", DIM)
    writer.add_uint32("lavasr.enhancer.ffn_dim", FFN_DIM)
    writer.add_uint32("lavasr.enhancer.n_blocks", N_BLOCKS)
    writer.add_uint32("lavasr.enhancer.n_mels", N_MELS)
    writer.add_uint32("lavasr.enhancer.kernel", KERNEL)
    writer.add_uint32("lavasr.enhancer.n_fft", N_FFT)
    writer.add_uint32("lavasr.enhancer.hop", HOP)
    writer.add_uint32("lavasr.enhancer.win", WIN)
    writer.add_uint32("lavasr.enhancer.spec_bins", SPEC_BINS)
    writer.add_uint32("lavasr.enhancer.mel_ref_sample_rate", MEL_REF_SR)
    writer.add_uint32("lavasr.enhancer.work_sample_rate", WORK_SR)
    writer.add_float32("lavasr.enhancer.clip_max", clip_max)
    writer.add_float32("lavasr.enhancer.layernorm_eps", LN_EPS)

    print("tensors:")
    # --- embed + first norm ---
    store(writer, "enhancer.embed.weight", bi["backbone.embed.weight"], args.ftype)
    store(writer, "enhancer.embed.bias", bi["backbone.embed.bias"], args.ftype, allow_f16=False)
    store(writer, "enhancer.norm.weight", bi["backbone.norm.weight"], args.ftype, allow_f16=False)
    store(writer, "enhancer.norm.bias", bi["backbone.norm.bias"], args.ftype, allow_f16=False)

    # --- 8 ConvNeXt blocks ---
    for i in range(N_BLOCKS):
        p = f"backbone.convnext.{i}"
        store(writer, f"enhancer.block.{i}.dwconv.weight", bi[f"{p}.dwconv.weight"], args.ftype)
        store(writer, f"enhancer.block.{i}.dwconv.bias", bi[f"{p}.dwconv.bias"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.norm.weight", bi[f"{p}.norm.weight"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.norm.bias", bi[f"{p}.norm.bias"], args.ftype, allow_f16=False)
        w1 = find_matmul_weight(bb, bi, bb_by_out, f"{p}.pwconv1.bias")  # [in=512, out=1536]
        w2 = find_matmul_weight(bb, bi, bb_by_out, f"{p}.pwconv2.bias")  # [in=1536, out=512]
        store(writer, f"enhancer.block.{i}.pwconv1.weight", w1.T, args.ftype)  # -> [out,in]
        store(writer, f"enhancer.block.{i}.pwconv1.bias", bi[f"{p}.pwconv1.bias"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.pwconv2.weight", w2.T, args.ftype)  # -> [out,in]
        store(writer, f"enhancer.block.{i}.pwconv2.bias", bi[f"{p}.pwconv2.bias"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.gamma", bi[f"{p}.gamma"], args.ftype, allow_f16=False)

    # --- final layer norm ---
    store(writer, "enhancer.final_norm.weight", bi["backbone.final_layer_norm.weight"], args.ftype, allow_f16=False)
    store(writer, "enhancer.final_norm.bias", bi["backbone.final_layer_norm.bias"], args.ftype, allow_f16=False)

    # --- spec head ---
    w_out = find_matmul_weight(sh, si, node_by_output(sh), "out.bias")  # [in=512, out=2050]
    store(writer, "spec_head.out.weight", w_out.T, args.ftype)  # -> [out=2050, in=512]
    store(writer, "spec_head.out.bias", si["out.bias"], args.ftype, allow_f16=False)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote {args.out} (arch={ARCH}, ftype={args.ftype}, clip_max={clip_max})")


if __name__ == "__main__":
    sys.exit(main())
