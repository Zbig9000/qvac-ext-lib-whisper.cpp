#!/usr/bin/env python3
"""Export the offline FastConformer encoder from a parakeet GGUF to a Core ML
package for the Apple Neural Engine sidecar consumed by the parakeet.cpp Engine
(the encoder I/O contract lives in src/coreml/parakeet-encoder.h).

The exported graph reuses the pure-PyTorch reference encoder in
ref-encoder-from-gguf.py so it matches the ggml encoder numerically. It is
traced at a fixed mel length (derived from a sample wav or passed explicitly),
since the offline sidecar binds to run_encoder's fixed-shape hand-off.

Example:

  python scripts/export-encoder-coreml.py \
      --gguf   models/parakeet-tdt-0.6b-v3.f16.gguf \
      --wav    test/samples/jfk.wav \
      --out    models/parakeet-tdt-0.6b-v3.f16-encoder.mlpackage \
      --compile-dir models
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


class EncoderModule(torch.nn.Module):
    def __init__(self, ref, weights, meta):
        super().__init__()
        self.ref = ref
        self.meta = meta
        self._keys = list(weights.keys())
        self._buffers_by_key = {}
        for i, key in enumerate(self._keys):
            name = f"w_{i}"
            self.register_buffer(name, weights[key].contiguous().float())
            self._buffers_by_key[key] = name

    def forward(self, mel):
        weights = BiasTolerantWeights(
            (key, getattr(self, self._buffers_by_key[key])) for key in self._keys)
        return encoder_forward(self.ref, mel, weights, self.meta)[0]


def compile_mlmodelc(mlpackage_path, compile_dir):
    subprocess.run(
        ["xcrun", "coremlc", "compile", str(mlpackage_path), str(compile_dir)],
        check=True)


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
    args = ap.parse_args()

    ref = load_reference_encoder(args.scripts)
    weights, meta = ref.load_gguf(args.gguf)

    d_model = meta["parakeet.encoder.d_model"]
    n_mels = int(weights["preproc.mel_filterbank"].shape[0])
    if args.n_mel_frames is not None:
        n_mel_frames = args.n_mel_frames
    else:
        n_mel_frames = mel_frames_for_wav(args.wav, resolve_hop_length(meta))

    model = EncoderModule(ref, weights, meta).eval()
    example = torch.zeros(n_mels, n_mel_frames, dtype=torch.float32)
    with torch.inference_mode():
        reference_out = model(example)
    print(f"[export] n_mels={n_mels} d_model={d_model} n_mel_frames={n_mel_frames} "
          f"encoder_frames={reference_out.shape[0]}")

    traced = torch.jit.trace(model, example, check_trace=False)
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    out_dtype = np.float16 if args.precision == "fp16" else np.float32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(n_mels, n_mel_frames), dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_out", dtype=out_dtype)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.save(str(args.out))
    print(f"[export] saved {args.out}")

    if args.compile_dir is not None:
        compile_mlmodelc(args.out, args.compile_dir)
        print(f"[export] compiled .mlmodelc into {args.compile_dir}")


if __name__ == "__main__":
    main()
