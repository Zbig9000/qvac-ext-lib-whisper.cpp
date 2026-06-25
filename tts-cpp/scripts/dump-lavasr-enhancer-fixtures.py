#!/usr/bin/env python3
"""Dump LavaSR enhancer fixtures for the C++ parity test (test-lavasr-enhancer-core).

Emits, into <out-dir>, one <canonical-name>.npy per enhancer weight tensor (in
the same orientation the GGUF / scalar core use) plus an onnxruntime golden:
  mel.npy   [80, T]     deterministic random log-mel input (seed 0)
  real.npy  [1025, T]   spec-head real output from the original ONNX graphs
  imag.npy  [1025, T]   spec-head imag output

Requires: numpy, onnx, onnxruntime.  Inputs are the public LavaSRcpp release
ONNX files (enhancer_backbone.onnx{,.data}, enhancer_spec_head.onnx{,.data}).

Usage:
  python dump-lavasr-enhancer-fixtures.py \
      --onnx-dir /path/to/lavasr/onnx \
      --out-dir  /path/to/lavasr/fixtures \
      --frames   50
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def by_out(g):
    d = {}
    for n in g.node:
        for o in n.output:
            d[o] = n
    return d


def matmul_weight(g, inits, bo, bias_name):
    for n in g.node:
        if n.op_type == "Add" and bias_name in n.input:
            other = [i for i in n.input if i != bias_name][0]
            for i in bo[other].input:
                if i in inits:
                    return inits[i]
    raise RuntimeError(f"no MatMul weight for {bias_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bb_path = os.path.join(args.onnx_dir, "enhancer_backbone.onnx")
    sh_path = os.path.join(args.onnx_dir, "enhancer_spec_head.onnx")
    bb = onnx.load(bb_path, load_external_data=True).graph
    sh = onnx.load(sh_path, load_external_data=True).graph
    bi = {t.name: numpy_helper.to_array(t) for t in bb.initializer}
    si = {t.name: numpy_helper.to_array(t) for t in sh.initializer}
    bb_bo, sh_bo = by_out(bb), by_out(sh)

    def save(name, arr):
        np.save(os.path.join(args.out_dir, name + ".npy"),
                np.ascontiguousarray(arr, dtype=np.float32))

    save("enhancer.embed.weight", bi["backbone.embed.weight"])
    save("enhancer.embed.bias", bi["backbone.embed.bias"])
    save("enhancer.norm.weight", bi["backbone.norm.weight"])
    save("enhancer.norm.bias", bi["backbone.norm.bias"])
    for i in range(8):
        p = f"backbone.convnext.{i}"
        save(f"enhancer.block.{i}.dwconv.weight", bi[f"{p}.dwconv.weight"])
        save(f"enhancer.block.{i}.dwconv.bias", bi[f"{p}.dwconv.bias"])
        save(f"enhancer.block.{i}.norm.weight", bi[f"{p}.norm.weight"])
        save(f"enhancer.block.{i}.norm.bias", bi[f"{p}.norm.bias"])
        # Linear weights are transposed to [out, in] to match the GGUF/core.
        save(f"enhancer.block.{i}.pwconv1.weight",
             matmul_weight(bb, bi, bb_bo, f"{p}.pwconv1.bias").T)
        save(f"enhancer.block.{i}.pwconv1.bias", bi[f"{p}.pwconv1.bias"])
        save(f"enhancer.block.{i}.pwconv2.weight",
             matmul_weight(bb, bi, bb_bo, f"{p}.pwconv2.bias").T)
        save(f"enhancer.block.{i}.pwconv2.bias", bi[f"{p}.pwconv2.bias"])
        save(f"enhancer.block.{i}.gamma", bi[f"{p}.gamma"])
    save("enhancer.final_norm.weight", bi["backbone.final_layer_norm.weight"])
    save("enhancer.final_norm.bias", bi["backbone.final_layer_norm.bias"])
    save("spec_head.out.weight", matmul_weight(sh, si, sh_bo, "out.bias").T)
    save("spec_head.out.bias", si["out.bias"])

    rng = np.random.RandomState(args.seed)
    T = args.frames
    mel = (rng.randn(80, T).astype(np.float32)) * 2.0 - 4.0
    bb_sess = ort.InferenceSession(bb_path, providers=["CPUExecutionProvider"])
    sh_sess = ort.InferenceSession(sh_path, providers=["CPUExecutionProvider"])
    hidden = bb_sess.run(None, {bb_sess.get_inputs()[0].name: mel[None]})[0]
    real, imag = sh_sess.run(None, {sh_sess.get_inputs()[0].name: hidden})
    save("mel", mel)
    save("real", real[0])
    save("imag", imag[0])
    print(f"Wrote enhancer fixtures to {args.out_dir} (T={T})")


if __name__ == "__main__":
    main()
