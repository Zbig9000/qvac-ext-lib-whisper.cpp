#!/usr/bin/env python3
"""Export the offline FastConformer encoder from a parakeet GGUF to a Core ML
package for the Apple Neural Engine sidecar consumed by the parakeet.cpp Engine
(the encoder I/O contract lives in src/coreml/parakeet-encoder.h).

The exported graph reuses the pure-PyTorch reference encoder in
ref-encoder-from-gguf.py so it matches the ggml encoder numerically.

Two input-shape modes:
  - Fixed (default): torch.jit.trace at a single mel length (from a sample wav
    or an explicit count). The sidecar then accelerates only utterances whose
    mel length equals that count; every other length falls back to ggml.
  - Flexible (--flexible): export a Core ML RangeDim time axis via torch.export
    dynamic shapes so one encoder serves variable-length utterances. Masking is
    dropped from the subsampling stack (a no-op for the full-valid offline
    input) and the relative positional table is baked as a fixed buffer, so the
    only length-dependent op left in the graph is a dynamic slice. This path is
    best-effort: it needs coremltools>=8 / torch>=2.3 and MUST be validated on
    Apple hardware (test-encoder-coreml-parity at several lengths + coreml-cli
    ANE op-placement), since torch.export / coremltools dynamic-shape support
    for the attention rel_shift can require model-specific tweaks.

Example:

  python scripts/export-encoder-coreml.py \
      --gguf   models/parakeet-tdt-0.6b-v3.f16.gguf \
      --wav    test/samples/jfk.wav \
      --out    models/parakeet-tdt-0.6b-v3-encoder.mlpackage \
      --compile-dir models

  # variable-length (best-effort; validate on device):
  python scripts/export-encoder-coreml.py \
      --gguf models/parakeet-tdt-0.6b-v3.f16.gguf --wav test/samples/jfk.wav \
      --flexible --max-frames 3000 \
      --out models/parakeet-tdt-0.6b-v3-encoder.mlpackage --compile-dir models
"""
import argparse
import importlib.util
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct


def load_reference_encoder(scripts_dir):
    path = scripts_dir / "ref-encoder-from-gguf.py"
    spec = importlib.util.spec_from_file_location("parakeet_ref_encoder", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BiasTolerantWeights(dict):
    def __missing__(self, key):
        if key.endswith(".bias"):
            return None
        raise KeyError(key)


def resolve_hop_length(meta):
    for key, value in meta.items():
        if "hop" in key.lower():
            return int(value)
    return 160


def mel_frames_for_wav(wav_path, hop_length):
    with wave.open(str(wav_path), "rb") as reader:
        num_samples = reader.getnframes()
    return 1 + num_samples // hop_length


def conformer_conv(x, weights, prefix):
    x = x.transpose(1, 2)
    x = F.conv1d(x, weights[f"{prefix}.pw1.weight"].squeeze(-1).unsqueeze(-1),
                 weights[f"{prefix}.pw1.bias"])
    x = F.glu(x, dim=1)
    depthwise = weights[f"{prefix}.dw.weight"]
    groups = int(depthwise.shape[0])
    padding = (int(depthwise.shape[-1]) - 1) // 2
    x = F.conv1d(x, depthwise, weights[f"{prefix}.dw.bias"], padding=padding, groups=groups)
    scale = weights[f"{prefix}.bn.scale"].view(1, -1, 1)
    shift = weights[f"{prefix}.bn.shift"].view(1, -1, 1)
    x = F.silu(x * scale + shift)
    x = F.conv1d(x, weights[f"{prefix}.pw2.weight"].squeeze(-1).unsqueeze(-1),
                 weights[f"{prefix}.pw2.bias"])
    return x.transpose(1, 2)


def conformer_block(ref, x, pos_emb, weights, index, n_heads):
    p = f"encoder.blk.{index}"
    x = x + 0.5 * ref.conformer_ff(
        ref.layer_norm(x, weights[f"{p}.norm_ff1.weight"], weights[f"{p}.norm_ff1.bias"]),
        weights, f"{p}.ff1")
    x = x + ref.rel_pos_mha(
        ref.layer_norm(x, weights[f"{p}.norm_attn.weight"], weights[f"{p}.norm_attn.bias"]),
        pos_emb, weights, f"{p}.attn", n_heads)
    x = x + conformer_conv(
        ref.layer_norm(x, weights[f"{p}.norm_conv.weight"], weights[f"{p}.norm_conv.bias"]),
        weights, f"{p}.conv")
    x = x + 0.5 * ref.conformer_ff(
        ref.layer_norm(x, weights[f"{p}.norm_ff2.weight"], weights[f"{p}.norm_ff2.bias"]),
        weights, f"{p}.ff2")
    return ref.layer_norm(x, weights[f"{p}.norm_out.weight"], weights[f"{p}.norm_out.bias"])


def encoder_forward(ref, mel, weights, meta):
    d_model = meta["parakeet.encoder.d_model"]
    n_layers = meta["parakeet.encoder.n_layers"]
    n_heads = meta["parakeet.encoder.n_heads"]
    x, _ = ref.subsampling(mel, weights)
    if meta.get("parakeet.encoder.xscaling", True):
        x = x * math.sqrt(d_model)
    length = x.size(1)
    pe = ref.sinusoidal_rel_pe(
        max(length, meta.get("parakeet.encoder.pos_emb_max_len", 5000)), d_model, dtype=x.dtype)
    center = pe.size(1) // 2 + 1
    pos_emb = pe[:, center - length: center + length - 1]
    for index in range(n_layers):
        x = conformer_block(ref, x, pos_emb, weights, index, n_heads)
    return x


def subsampling_shape_generic(mel, weights):
    # ref.subsampling masks each conv stage to a valid length derived from
    # torch.arange(T), which bakes the traced mel length into the graph. For the
    # full-valid offline input those masks are all-ones (a no-op), so drop them
    # here to keep the stride-2 conv stack polymorphic in the time axis -- the
    # prerequisite for a dynamic-length (RangeDim) export.
    x = mel.unsqueeze(0).transpose(1, 2).unsqueeze(1)
    x = F.conv2d(x, weights["encoder.subsampling.conv0.weight"],
                 bias=weights["encoder.subsampling.conv0.bias"], stride=2, padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights["encoder.subsampling.conv1_dw.weight"],
                 bias=weights["encoder.subsampling.conv1_dw.bias"],
                 stride=2, padding=1, groups=x.size(1))
    x = F.conv2d(x, weights["encoder.subsampling.conv1_pw.weight"],
                 bias=weights["encoder.subsampling.conv1_pw.bias"], stride=1, padding=0)
    x = F.relu(x)
    x = F.conv2d(x, weights["encoder.subsampling.conv2_dw.weight"],
                 bias=weights["encoder.subsampling.conv2_dw.bias"],
                 stride=2, padding=1, groups=x.size(1))
    x = F.conv2d(x, weights["encoder.subsampling.conv2_pw.weight"],
                 bias=weights["encoder.subsampling.conv2_pw.bias"], stride=1, padding=0)
    x = F.relu(x)
    x = x.permute(0, 2, 1, 3).flatten(2)
    return F.linear(x, weights["encoder.subsampling.out.weight"],
                    weights["encoder.subsampling.out.bias"])


def encoder_frames_for_mel(ref, meta, n_mel_frames):
    # Post-subsampling encoder frame count: the same three stride-2 convs
    # run_encoder uses, mirrored here to size the fixed positional-embedding
    # buffer to the longest supported utterance.
    causal = meta.get("parakeet.encoder.causal_downsampling", False)

    def nxt(length):
        return (length // 2 + 1) if causal else ref._conv_out_len(length, 3, 2, 1)

    return nxt(nxt(nxt(n_mel_frames)))


def encoder_forward_flexible(ref, mel, weights, meta, pe_table):
    # Variable-length forward: masking-free subsampling + a precomputed relative
    # positional buffer sliced by the symbolic sequence length, so the only
    # length-dependent op left in the graph is the dynamic slice.
    d_model = meta["parakeet.encoder.d_model"]
    n_layers = meta["parakeet.encoder.n_layers"]
    n_heads = meta["parakeet.encoder.n_heads"]
    x = subsampling_shape_generic(mel, weights)
    if meta.get("parakeet.encoder.xscaling", True):
        x = x * math.sqrt(d_model)
    length = x.size(1)
    center = pe_table.size(1) // 2 + 1
    pos_emb = pe_table[:, center - length: center + length - 1]
    for index in range(n_layers):
        x = conformer_block(ref, x, pos_emb, weights, index, n_heads)
    return x


class EncoderModule(torch.nn.Module):
    def __init__(self, ref, weights, meta, flexible=False, pe_table=None):
        super().__init__()
        self.ref = ref
        self.meta = meta
        self.flexible = flexible
        self._keys = list(weights.keys())
        self._buffers_by_key = {}
        for i, key in enumerate(self._keys):
            name = f"w_{i}"
            self.register_buffer(name, weights[key].contiguous().float())
            self._buffers_by_key[key] = name
        if pe_table is not None:
            self.register_buffer("pe_table", pe_table.contiguous().float())

    def forward(self, mel):
        weights = BiasTolerantWeights(
            (key, getattr(self, self._buffers_by_key[key])) for key in self._keys)
        if self.flexible:
            return encoder_forward_flexible(self.ref, mel, weights, self.meta, self.pe_table)[0]
        return encoder_forward(self.ref, mel, weights, self.meta)[0]


def compile_mlmodelc(mlpackage_path, compile_dir):
    subprocess.run(
        ["xcrun", "coremlc", "compile", str(mlpackage_path), str(compile_dir)],
        check=True)


def convert_fixed(ref, weights, meta, example, n_mels, n_mel_frames, d_model,
                  precision, out_dtype):
    model = EncoderModule(ref, weights, meta).eval()
    with torch.inference_mode():
        reference_out = model(example)
    print(f"[export] fixed n_mels={n_mels} d_model={d_model} n_mel_frames={n_mel_frames} "
          f"encoder_frames={reference_out.shape[0]}")
    traced = torch.jit.trace(model, example, check_trace=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(n_mels, n_mel_frames), dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_out", dtype=out_dtype)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS13,
    )


def convert_flexible(ref, weights, meta, example, n_mels, n_mel_frames, d_model,
                     min_frames, max_frames, precision, out_dtype):
    min_frames = min_frames if min_frames is not None else 1
    max_frames = max(max_frames if max_frames is not None else n_mel_frames, n_mel_frames)
    pos_emb_max_len = int(meta.get("parakeet.encoder.pos_emb_max_len", 5000))
    pe_len = max(pos_emb_max_len, encoder_frames_for_mel(ref, meta, max_frames))
    pe_table = ref.sinusoidal_rel_pe(pe_len, d_model)

    model = EncoderModule(ref, weights, meta, flexible=True, pe_table=pe_table).eval()
    with torch.inference_mode():
        reference_out = model(example)
    print(f"[export] flexible n_mels={n_mels} d_model={d_model} "
          f"mel_frames=[{min_frames},{max_frames}] traced={n_mel_frames} "
          f"encoder_frames={reference_out.shape[0]} pe_len={pe_len}")

    # torch.export keeps the time axis symbolic (unlike jit.trace, which bakes it),
    # so the dynamic pos-emb slice survives conversion. strict=False uses the
    # non-strict tracer, which tolerates the Python-level weights dict; flip it if a
    # given torch version prefers strict capture.
    time_dim = torch.export.Dim("mel_t", min=min_frames, max=max_frames)
    exported = torch.export.export(
        model, (example,), dynamic_shapes={"mel": {1: time_dim}}, strict=False)
    return ct.convert(
        exported,
        inputs=[ct.TensorType(
            name="mel",
            shape=ct.Shape(shape=(n_mels, ct.RangeDim(
                lower_bound=min_frames, upper_bound=max_frames, default=n_mel_frames))),
            dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_out", dtype=out_dtype)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS13,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output .mlpackage path")
    ap.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=Path, help="size the encoder to this wav's mel length")
    group.add_argument("--n-mel-frames", type=int, help="fixed mel length to trace")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--compile-dir", type=Path, default=None,
                    help="if set, compile the .mlpackage to a .mlmodelc here via coremlc")
    ap.add_argument("--flexible", action="store_true",
                    help="export a variable-length (Core ML RangeDim) encoder via "
                         "torch.export; best-effort, needs coremltools>=8 / torch>=2.3 and "
                         "on-device (ANE) parity validation")
    ap.add_argument("--min-frames", type=int, default=None,
                    help="[--flexible] minimum mel length the RangeDim accepts (default 1)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="[--flexible] maximum mel length the RangeDim accepts; set to your "
                         "longest expected utterance (default: the traced length)")
    args = ap.parse_args()

    ref = load_reference_encoder(args.scripts)
    weights, meta = ref.load_gguf(args.gguf)

    d_model = meta["parakeet.encoder.d_model"]
    n_mels = int(weights["preproc.mel_filterbank"].shape[0])
    if args.n_mel_frames is not None:
        n_mel_frames = args.n_mel_frames
    else:
        n_mel_frames = mel_frames_for_wav(args.wav, resolve_hop_length(meta))

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    out_dtype = np.float16 if args.precision == "fp16" else np.float32
    example = torch.zeros(n_mels, n_mel_frames, dtype=torch.float32)

    if args.flexible:
        mlmodel = convert_flexible(ref, weights, meta, example, n_mels, n_mel_frames,
                                   d_model, args.min_frames, args.max_frames,
                                   precision, out_dtype)
    else:
        mlmodel = convert_fixed(ref, weights, meta, example, n_mels, n_mel_frames,
                                d_model, precision, out_dtype)
    mlmodel.save(str(args.out))
    print(f"[export] saved {args.out}")

    if args.compile_dir is not None:
        compile_mlmodelc(args.out, args.compile_dir)
        print(f"[export] compiled .mlmodelc into {args.compile_dir}")


if __name__ == "__main__":
    main()
