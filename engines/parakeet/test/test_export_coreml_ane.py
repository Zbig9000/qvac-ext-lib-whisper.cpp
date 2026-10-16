#!/usr/bin/env python3
"""Model-free checks for the Neural Engine Core ML exporters.

Covers the pure helpers of export-encoder-coreml-ane.py (frame arithmetic and
the channel-axis LayerNorm the ANE graph uses) and the TDT decoder graph of
export-decoder-coreml-ane.py against its scalar reference on random weights.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None

if torch is None:
    print("SKIP: torch is not installed")
    raise SystemExit(0)

if "coremltools" not in sys.modules:
    try:
        import coremltools  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["coremltools"] = types.ModuleType("coremltools")

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER = load("export-encoder-coreml-ane.py", "export_encoder_coreml_ane")
DECODER = load("export-decoder-coreml-ane.py", "export_decoder_coreml_ane")


class EncoderHelperTests(unittest.TestCase):
    def test_default_window_is_fifteen_seconds(self):
        self.assertEqual(ENCODER.DEFAULT_WINDOW_MEL_FRAMES, 1500)

    def test_encoder_frames_for_offline_window(self):
        # Three stride-2 convs, kernel 3, pad 1: 1500 -> 750 -> 375 -> 188.
        self.assertEqual(ENCODER.encoder_frames_for_mel(1500, causal=False), 188)
        self.assertEqual(ENCODER.encoder_frames_for_mel(1501, causal=False), 188)
        self.assertEqual(ENCODER.encoder_frames_for_mel(326, causal=False), 41)

    def test_encoder_frames_for_causal_window(self):
        self.assertEqual(ENCODER.encoder_frames_for_mel(1500, causal=True), 189)

    def test_channel_norm_matches_layer_norm_over_channels(self):
        torch.manual_seed(0)
        x = torch.randn(1, 6, 1, 5)
        gamma = torch.randn(1, 6, 1, 1)
        beta = torch.randn(1, 6, 1, 1)
        got = ENCODER.channel_norm(x, gamma, beta, eps=1e-5)
        want = F.layer_norm(x.permute(0, 2, 3, 1), (6,), gamma.reshape(6), beta.reshape(6),
                            eps=1e-5).permute(0, 3, 1, 2)
        self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_key_bias_pad_is_finite(self):
        self.assertTrue(np.isfinite(np.float16(ENCODER.KEY_BIAS_PAD)))


def toy_tdt_weights(h_pred=4, h_joint=3, d_enc=5, vocab=6, n_durations=3, layers=2):
    torch.manual_seed(1)
    weights = {}
    for layer in range(layers):
        p = f"tdt.predict.lstm.{layer}"
        weights[f"{p}.w_ih"] = torch.randn(4 * h_pred, h_pred) * 0.3
        weights[f"{p}.w_hh"] = torch.randn(4 * h_pred, h_pred) * 0.3
        weights[f"{p}.b_ih"] = torch.randn(4 * h_pred) * 0.1
        weights[f"{p}.b_hh"] = torch.randn(4 * h_pred) * 0.1
    weights["tdt.predict.embed.weight"] = torch.randn(vocab + 1, h_pred)
    weights["tdt.joint.enc.weight"] = torch.randn(h_joint, d_enc) * 0.3
    weights["tdt.joint.enc.bias"] = torch.randn(h_joint) * 0.1
    weights["tdt.joint.pred.weight"] = torch.randn(h_joint, h_pred) * 0.3
    weights["tdt.joint.pred.bias"] = torch.randn(h_joint) * 0.1
    weights["tdt.joint.out.weight"] = torch.randn(vocab + 1 + n_durations, h_joint) * 0.3
    weights["tdt.joint.out.bias"] = torch.randn(vocab + 1 + n_durations) * 0.1
    meta = {
        "parakeet.tdt.pred_rnn_layers": layers,
        "parakeet.tdt.pred_hidden": h_pred,
        "parakeet.tdt.joint_hidden": h_joint,
        "parakeet.tdt.vocab_size": vocab,
        "parakeet.tdt.blank_id": vocab,
        "parakeet.tdt.num_durations": n_durations,
    }
    return weights, meta


class DecoderExporterTests(unittest.TestCase):
    def test_geometry_reads_the_checkpoint(self):
        weights, meta = toy_tdt_weights()
        geom = DECODER.TdtGeometry(weights, meta)
        self.assertEqual((geom.n_layers, geom.h_pred, geom.d_enc, geom.v_out), (2, 4, 5, 10))
        self.assertEqual(geom.blank_id, 6)

    def test_geometry_rejects_wrong_joint_output(self):
        weights, meta = toy_tdt_weights()
        meta["parakeet.tdt.num_durations"] = 2
        with self.assertRaises(RuntimeError):
            DECODER.TdtGeometry(weights, meta)

    def test_geometry_rejects_missing_lstm_layer(self):
        weights, meta = toy_tdt_weights()
        meta["parakeet.tdt.pred_rnn_layers"] = 3
        with self.assertRaises(RuntimeError):
            DECODER.TdtGeometry(weights, meta)

    def test_graph_matches_scalar_reference(self):
        weights, meta = toy_tdt_weights()
        geom = DECODER.TdtGeometry(weights, meta)
        model = DECODER.AneTdtDecoder(weights, geom, lanes=3, width=1).eval()
        self.assertTrue(DECODER.check_parity(model, weights, geom))

    def test_default_lanes_and_width(self):
        self.assertEqual(DECODER.DEFAULT_LANES, 16)
        self.assertEqual(DECODER.DEFAULT_WIDTH, 1)


if __name__ == "__main__":
    unittest.main()
