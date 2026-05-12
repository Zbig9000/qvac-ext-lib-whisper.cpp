# Supertonic → ggml Port: Development Journal

This document tracks the experimental **Supertonic / Supertonic 2** GGUF +
GGML runtime added to this repo: what was tested, what matched, what sounded
good, which performance ideas worked, and which optimization attempts were
rolled back or deferred.

It is separate from `PROGRESS.md`, which covers the Chatterbox Turbo and
Chatterbox Multilingual ports.  Supertonic is a different architecture and is
currently implemented as a model-specific runtime over official ONNX weights
converted into one GGUF.

- **Models**:
  - `Supertone/supertonic` — stable English bundle, no language wrapping.
  - `Supertone/supertonic-2` — multilingual bundle, open/close language tags
    (`<lang>text</lang>`).
- **Goal**: run the known Supertonic stages in C++/GGML with numerical parity
  against ONNX Runtime, clean audio output, and production-grade CPU
  performance.
- **Final CPU benchmark target**: matched GGML vs ONNX Runtime
  `CPUExecutionProvider` at 1, 2, 3, and 4 threads.

---

## Current Status

The branch now contains a full Supertonic path:

| Binary / script | Role |
|---|---|
| `scripts/setup-supertonic2.sh` | Downloads the official Hugging Face bundle and writes the local GGUF. |
| `scripts/convert-supertonic2-to-gguf.py` | Converts official ONNX/assets into `models/supertonic2.gguf` or `models/supertonic.gguf`. |
| `build/tts-cli` | Autodetects `supertonic.arch` and routes Supertonic text → 44.1 kHz wav on CPU. |
| `build/supertonic-cli` | Focused Supertonic compatibility/debug wrapper. |
| `build/supertonic-bench` | Per-stage Supertonic benchmark with JSON output. |
| `test-supertonic-*` | Stage and trace parity harnesses against ONNX reference dumps. |

The generated GGUF files are intentionally not committed:

```text
models/supertonic.gguf   ~250 MB
models/supertonic2.gguf  ~251 MB
```

They are ignored by `.gitignore` (`models/`, `*.gguf`), matching the existing
Chatterbox approach where converters/setup scripts create local model files.

### Correctness

The full path is implemented, and all model stages are routed through the
GGML-backed production path:

1. preprocess
2. duration predictor
3. text encoder
4. vector estimator
5. vocoder

The end-to-end pipeline parity check against the Supertonic 2 ONNX reference
passes:

| Check | Result |
|---|---:|
| `test-supertonic-pipeline` max abs | `3.431e-05` |
| `test-supertonic-pipeline` RMS | `2.086e-06` |
| vocoder pointwise harness | PASS |

Audio checks were clean for generated English, French, and Portuguese samples.

### Final CPU Benchmark

Final benchmark settings:

- GGML: `models/supertonic2.gguf`
- ONNX: official Supertonic 2 ONNX files via ONNX Runtime
  `CPUExecutionProvider`
- Voice: `F1`
- Steps: `5`
- Speed: `1.05`
- Runs: `3`, warmup: `1`
- Prompts: quick English, longer English, Portuguese smoke
- Thread matrix: 1v1, 2v2, 3v3, 4v4

Median total wall time in milliseconds:

| Prompt | GGML 1t | GGML 2t | GGML 3t | GGML 4t | ONNX 1t | ONNX 2t | ONNX 3t | ONNX 4t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quick English | 298.0 | 189.4 | 157.7 | 157.7 | 373.8 | 218.5 | 168.3 | 148.8 |
| longer English | 757.5 | 491.2 | 390.3 | 361.2 | 1103.0 | 580.6 | 555.7 | 351.5 |
| Portuguese smoke | 457.2 | 292.9 | 251.0 | 234.3 | 610.6 | 344.6 | 268.3 | 250.8 |

Headline:

- GGML wins **10 / 12** matched comparisons.
- GGML wins **all 1-thread** comparisons.
- GGML vocoder wins the 4-thread stage comparison on all tested prompts.
- Remaining losses are narrow:
  - quick English 4t: GGML `157.7 ms` vs ONNX `148.8 ms`
  - longer English 4t: GGML `361.2 ms` vs ONNX `351.5 ms`

4-thread stage medians:

| Prompt | Runtime | Duration | Text | Vector | Vocoder | Total |
|---|---|---:|---:|---:|---:|---:|
| quick English | GGML | 3.9 | 13.5 | 96.3 | 43.6 | 157.7 |
| quick English | ONNX | 1.5 | 11.5 | 85.9 | 49.8 | 148.8 |
| longer English | GGML | 11.9 | 33.3 | 201.2 | 115.1 | 361.2 |
| longer English | ONNX | 2.4 | 13.1 | 198.3 | 138.8 | 351.5 |
| Portuguese smoke | GGML | 6.5 | 20.8 | 137.6 | 68.9 | 234.3 |
| Portuguese smoke | ONNX | 1.7 | 11.6 | 141.7 | 95.6 | 250.8 |

---

## Repository Additions

```text
include/tts-cpp/supertonic/engine.h       public Supertonic synth API
scripts/convert-supertonic2-to-gguf.py    ONNX/assets → Supertonic GGUF
scripts/setup-supertonic2.sh              download + convert wrapper
scripts/dump-supertonic-reference.py      ONNX reference tensor dumper
scripts/bench-supertonic-onnx.py          ONNX Runtime benchmark script
src/supertonic_gguf.cpp                   GGUF loader + backend/thread setup
src/supertonic_preprocess.cpp             Unicode/text preprocessing
src/supertonic_duration.cpp               duration predictor
src/supertonic_text_encoder.cpp           text encoder
src/supertonic_vector_estimator.cpp       vector denoiser
src/supertonic_vocoder.cpp                vocoder
src/supertonic_engine.cpp                 end-to-end Supertonic API
src/supertonic_cli.cpp                    standalone Supertonic CLI
src/supertonic_bench.cpp                  GGML benchmark harness
src/test_supertonic_*.cpp                 stage parity and trace tests
```

---

## Development Log

### 1. Scoping: ONNX → GGUF is feasible, generic ONNX execution is not needed

The first decision was to avoid a generic ONNX executor.  Supertonic has four
known ONNX submodels plus stable assets (`tts.json`, `unicode_indexer.json`,
voice styles).  That makes a model-specific converter and model-specific C++
runtime the right shape.

The GGUF stores:

- all ONNX initializers
- tensor-valued ONNX constants
- `tts.json` metadata
- Unicode indexer
- built-in voice styles
- arrays mapping short GGUF tensor names back to the original ONNX source names

This source-name mapping was important.  Some ONNX tensor names are long or not
pleasant as ggml tensor names, but the C++ runtime can still request weights by
their original source names.

### 2. Early audio finding: stutter was language wrapping, not GGUF

The first audible issue was English stuttering / mechanical audio in
Supertonic 2.  The root cause was not the C++ port or GGUF conversion.

What failed:

- Old Supertonic 2 prefix-only wrapping:

```text
<en>text 
```

What worked:

- Stable English bundle (`Supertone/supertonic`) with no wrapping.
- Supertonic 2 multilingual bundle with open/close wrapping:

```text
<en>text</en>
<pt>text</pt>
<fr>text</fr>
```

This is now encoded in GGUF metadata as `supertonic.language_wrap_mode`, and the
runtime follows the metadata.

### 3. Reference and parity harnesses

Added ONNX reference dump scripts and stage tests before optimizing.  This was
essential because several later "obvious" graph fusions produced valid-looking
output tensors with bad data.

Useful parity tools:

- `test-supertonic-preprocess`
- `test-supertonic-duration`
- `test-supertonic-duration-trace`
- `test-supertonic-text-encoder`
- `test-supertonic-text-encoder-trace`
- `test-supertonic-vector`
- `test-supertonic-vector-trace`
- `test-supertonic-vocoder`
- `test-supertonic-vocoder-trace`
- `test-supertonic-vocoder-pointwise`
- `test-supertonic-pipeline`

Important reproducibility fix:

- C++ `std::normal_distribution` does not match NumPy's `RandomState`.
- The runtime now uses a NumPy-compatible MT19937 + `standard_normal()` path so
  `--seed 42` matches the ONNX/Python reference noise behavior.

### 4. Baseline: scalar C++ proved correctness but was far behind ONNX

The first full C++ path was useful for parity but not performance.

Initial scalar-era benchmark on the quick prompt showed roughly:

| Stage | ONNX | early C++ |
|---|---:|---:|
| duration | 1.72 ms | 8.28 ms |
| text encoder | 9.33 ms | 211.97 ms |
| vector estimator | 99.90 ms | 7156.24 ms |
| vocoder | 69.03 ms | 7080.52 ms |
| total | 180.32 ms | 14451.06 ms |

This made the priority clear: vector estimator and vocoder dominated the wall
time, followed by the text encoder.

### 5. Production controls: threading and BLAS policy

What worked:

- Add `supertonic_set_n_threads()`.
- Route all graph execution through `supertonic_graph_compute()`.
- Set CPU backend thread count before graph compute.
- Cap default thread count at 4 for the current small-graph Supertonic path.
- Cap BLAS worker threads by default:
  - `VECLIB_MAXIMUM_THREADS=1` on Accelerate
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `BLIS_NUM_THREADS=1`

Why this mattered:

The Supertonic CPU runtime already parallelizes work through GGML tasking and
custom-op task splits.  Letting BLAS also spawn worker pools for every small
pointwise matmul hurt 3-4 thread scaling.

### 6. Text encoder optimization

What worked:

- Move the text encoder production path to GGML.
- Express text ConvNeXt blocks in GGML.
- Use `ggml_flash_attn_ext` for speech-prompted attention.
- Implement relative-position self-attention with stock GGML ops.
- Cache relative-position attention graphs (`text_relpos_graph_cache`).
- Move FFN blocks from scalar C++ loops to cached GGML graphs.
- Refactor Q/K/V projections so outputs are closer to the needed channel-major
  layout and avoid some reshape/permute/contiguous overhead.

What did not get implemented yet:

- A custom fused relpos attention op.

Why it was deferred:

Profiling showed stock-op relpos was around `0.7-0.8 ms/layer` on the quick
prompt after the cached graph/FFN work.  That is not free, but the bigger
performance opportunities were still in vector/vocoder and graph boundary
overhead.

### 7. Vector estimator optimization

The vector estimator was the largest and most complicated optimization target.
It runs multiple attention and ConvNeXt-style groups per denoise step, then
repeats for the configured number of steps.

What worked:

- Split trace and production paths so production no longer scans debug trace
  vectors.
- Cache host-side static layout conversions for text embeddings and style
  contexts.
- Split text attention into QKV projection and attention-only cached graphs.
- Split style attention similarly.
- Reuse attention-only graph states for text and style attention.
- Replace D/L/H host packing with strided GGML views where layout allows it.
- Add persistent graph/allocr caches for vector attention, group, and tail
  islands.
- Gate intermediate graph outputs with `trace_outputs=false` in production.
- Fuse ConvNeXt group output with following text-attention QKV projection.
- Fuse residual/post-ConvNeXt boundaries with following style QKV projection.
- Fuse tail projection/update into a custom production op.
- Replace graph transpose-heavy dense time matmul with a direct BLAS custom op.
- Fuse ConvNeXt elementwise work:
  - `pw1 bias + GELU`
  - `pw2 bias + gamma + residual`

Portable custom CPU kernels added:

- K=1 pointwise Conv1D, BLAS/Accelerate-backed.
- K=5 depthwise Conv1D custom op with unrolled hot path.
- General fallback for other depthwise kernels.
- Direct row-wise layer norm.
- Direct dense time matmul.
- Tail update fusion.

What failed or was rolled back:

| Attempt | Result |
|---|---|
| Fold style residuals directly into attention graphs | Rolled back. Trace showed in-graph residual add corrupted the left-hand activation, likely due to GGML buffer lifetime / aliasing. |
| Temporary reusable D/L/H host packing buffers | Helped but was superseded by strided GGML views, which avoid the packing entirely where possible. |
| Broad graph folding without parity trace boundaries | Too risky. The vector trace harness showed small-looking graph rewrites can corrupt later residual paths. |

Main remaining vector issue:

- At higher thread counts, vector is close to ONNX but still has some variance.
- The next target should be graph scheduling/scaling stability, not a broad
  rewrite.

### 8. Vocoder optimization

The vocoder started as one of the two massive scalar bottlenecks.

What worked:

- Convert vocoder execution to a persistent GGML graph cache.
- Add a vocoder pointwise harness to isolate weight layout, BLAS layout, and
  custom-op parity.
- Use BLAS/Accelerate-backed K=1 causal Conv1D for hot projection paths.
- Use BLAS-backed K>1 causal Conv1D for `head1`.
- Keep the rest of the graph stable and parity-checked.

What failed:

| Attempt | Result |
|---|---|
| Broad K=1 BLAS replacement across vocoder too early | Failed parity until layout and tasking were isolated. |
| Custom op running BLAS work on every GGML task | Race / concurrent writes. Fixed by only doing the BLAS call on `ith == 0` for those ops. |
| Wrong transpose assumption for Conv1D weights | Produced large errors. The pointwise harness confirmed the correct `blas_col_nn` mapping. |

Final important point:

The vocoder is no longer the bottleneck.  In the final 4-thread comparison,
GGML vocoder beats ONNX on all three tested prompts.

### 9. Benchmark tooling

Added machine-readable benchmark output on both sides:

- `supertonic-bench --json-out`
- `scripts/bench-supertonic-onnx.py --json-out`
- `scripts/bench-supertonic-onnx.py --providers CPUExecutionProvider`
- `scripts/bench-supertonic-onnx.py --threads`
- `scripts/bench-supertonic-onnx.py --language-wrap-mode open_close`

This avoided a repeated source of confusion: ONNX and GGML must use the same
language wrapping, prompt, voice, steps, speed, thread count, and CPU provider.

Final matrix artifacts were written under:

```text
artifacts/supertonic-thread-matrix/
```

That directory is intentionally ignored.

### 10. Setup and local model workflow

The GGUF is not committed.  The repo now follows the Chatterbox pattern:

- converters/setup scripts create the local model
- runtime stays network-free
- missing model errors point users to setup commands

Common setup:

```bash
# Multilingual Supertonic 2
bash scripts/setup-supertonic2.sh

# Stable English Supertonic
bash scripts/setup-supertonic2.sh --arch supertonic
```

The lower-level converter also supports local ONNX assets:

```bash
python scripts/convert-supertonic2-to-gguf.py \
  --onnx-dir /path/to/supertonic-pytorch/onnx_models/onnx \
  --assets-dir /path/to/supertonic-pytorch/assets \
  --out models/supertonic2.gguf \
  --validate
```

---

## What Worked Best

1. **Parity-first development.**

   The trace harnesses caught layout bugs and graph aliasing failures that would
   otherwise have shown up only as bad audio.

2. **Model-specific GGUF, not generic ONNX execution.**

   Supertonic's stage boundaries are stable enough that a dedicated converter
   and runtime are simpler and faster.

3. **Open/close language wrapping for Supertonic 2.**

   This solved the English stutter without changing model math.

4. **Persistent GGML graph/allocr caches.**

   Reusing graph structure was essential for small repeated vector/text islands.

5. **Strided attention views.**

   Avoiding host D/L/H packing reduced repeated layout overhead and better
   matches the Chatterbox-style GGML approach.

6. **Targeted portable custom CPU kernels.**

   Pointwise Conv1D, depthwise Conv1D, row-wise layer norm, and dense time
   matmul were the right level of specialization: portable C++/CBLAS/Accelerate
   without locking the runtime to one CPU vendor.

7. **BLAS thread caps.**

   Preventing nested thread pools improved scaling stability.

8. **The isolated vocoder pointwise harness.**

   It quickly separated weight-layout bugs from GGML custom-op scheduling bugs.

---

## What Did Not Work

1. **Assuming ONNX/PyTorch reconstruction quality represented the official path.**

   The unofficial PyTorch reconstruction was useful for exploration but not a
   reliable audio-quality source.  Official ONNX assets plus correct wrapping
   were the right reference.

2. **Prefix-only language tags for Supertonic 2 English.**

   This caused audible stutter.  Use no wrapping for stable English
   `Supertone/supertonic`, and open/close wrapping for Supertonic 2.

3. **Folding graph boundaries before proving alias safety.**

   A style residual fold corrupted activations due to GGML buffer aliasing risk.
   Graph fusion must be guarded by trace parity.

4. **Broad custom-kernel rollout without isolated harnesses.**

   The vocoder K=1 BLAS path only became reliable after the isolated pointwise
   harness proved the exact tensor/BLAS layout.

5. **Letting BLAS and GGML both freely multi-thread.**

   Nested thread pools hurt the small-island workload.

6. **Trying to optimize only for Apple Accelerate.**

   The final custom kernels were kept portable: Accelerate where available,
   generic CBLAS elsewhere, and scalar fallbacks for unsupported cases.

---

## GPU bring-up: OpenCL (May 2026)

Target: the same `--n-gpu-layers > 0` flag already exposed by the
Supertonic CLI, but resolved to **OpenCL** instead of falling back to
CPU.  Tracking ticket: QVAC-18607.

### What was missing

The Supertonic CPU path (§7-§8 above) earned its CPU benchmark wins by
moving every hot loop onto a `ggml_custom_4d` op whose callback runs
CBLAS / pointer-arithmetic directly against the tensor `data` field:

| TU | Custom ops |
|----|-----------|
| `supertonic_vocoder.cpp` | K=1 cblas conv1d, K>1 cblas conv1d, depthwise dilated conv1d |
| `supertonic_vector_estimator.cpp` | conv1d_f32(K=1), depthwise same-padded conv1d, row-wise layer-norm, dense-time matmul, fused bias+GELU, fused (pw2 bias + γ + residual), fused tail-update (BLAS GEMM + mask + step-scale + residual add) |

None of those callbacks are valid on a GPU backend: `GGML_OP_CUSTOM`
isn't supported by `ggml-opencl` (or by CUDA / Metal / Vulkan), and the
op callbacks themselves assume host-addressable `data` pointers that
no GPU backend exposes inside graph execution.  So before this round,
loading Supertonic with `--n-gpu-layers > 0` either fell straight back
to CPU via `init_supertonic_backend` (when the backend wasn't compiled
in) or asserted at `ggml_backend_graph_compute` time inside the OpenCL
dispatch loop (when it was).

In addition, two builtins in the vocoder graph had similar portability
holes against baseline upstream OpenCL: `ggml_leaky_relu`
(`GGML_OP_LEAKY_RELU`) is only present on `ggml-opencl` builds that
carry the chatterbox `ggml-opencl-chatterbox-ops.patch` — fine for the
QVAC `ggml-speech` vcpkg consumption path, but unsafe for any other
GPU backend wanting Supertonic.

### What landed

| Change | File(s) |
|--------|---------|
| `supertonic_model::backend_is_cpu` set from `ggml_backend_is_cpu(model.backend)` right after `init_supertonic_backend()` resolves the device. | `supertonic_gguf.cpp`, `supertonic_internal.h` |
| `supertonic_op_dispatch_scope` — thread-local RAII helper instantiated at every public `supertonic_*_forward_ggml` / `*_trace_ggml` entry point.  Mirrors `model.backend_is_cpu` and `model.use_f16_attn` into the two thread-local flags consulted by the graph-build helpers. | `supertonic_internal.h`, `supertonic_gguf.cpp`, `supertonic_vocoder.cpp`, `supertonic_vector_estimator.cpp`, `supertonic_text_encoder.cpp`, `supertonic_duration.cpp` |
| Every `ggml_custom_4d` site gated on `supertonic_use_cpu_custom_ops()` so GPU runs fall through to the existing pure-GGML paths (`ggml_im2col + ggml_mul_mat`, `ggml_norm`, etc.) — all of which `ggml-opencl` already supports natively (see `ggml_opencl_supports_op()` in `ggml/src/ggml-opencl/ggml-opencl.cpp`). | `supertonic_vocoder.cpp`, `supertonic_vector_estimator.cpp` |
| Portable `leaky_relu_portable_ggml()` helper: on CPU keeps the fused builtin; on GPU decomposes into `RELU + SCALE + ADD`, all universally supported. | `supertonic_vocoder.cpp` |

### Optimization #1: F16 K/V flash-attention

The vector estimator's text-conditioned attention runs four times per
denoising step × N steps, so it's the single hottest op in the
Supertonic synthesis budget after the dense convnext blocks.  Lifted
straight from chatterbox's Adreno bring-up (§ `OpenCL optimization
log`), the vector-estimator graph now optionally materialises K / V
into contiguous F16 before calling `ggml_flash_attn_ext`, which makes
OpenCL dispatch the `flash_attn_f32_f16` kernel instead of the
F32-only one.  In chatterbox's Q4_0 CFM smoke run this dropped the
attention kernel from `~257 ms` to `~102 ms` on Adreno 830.

- Engine option: `EngineOptions::f16_attn` (`-1`=auto, `0`=off, `1`=on).
  Auto-enables on GPU backends, off on CPU.
- CLI flag: `--f16-attn 0|1`, exposed on `tts-cli`, `supertonic-cli`,
  and `supertonic-bench`.
- Cache key: `vector_text_attention_cache::f16_kv_attn` so toggling the
  flag mid-process safely rebuilds the cached graph.

Q stays F32: cheaper to keep one operand at the higher precision than
to round-trip the post-attention output back through F32 for the
downstream dense projection.

### How to use

```bash
# Build with OpenCL (in the standalone tree; in-tree subtree consumes
# ggml-speech vcpkg port which already carries the OpenCL patches).
cmake -S . -B build-opencl -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON
cmake --build build-opencl -j$(nproc) --target tts-cli supertonic-bench

# Run on OpenCL with auto F16 attention.
./build-opencl/supertonic-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --n-gpu-layers 99 \
  --out /tmp/supertonic2.wav

# Force F16 attention off (CPU-style fallback) for parity:
./build-opencl/supertonic-cli ... --n-gpu-layers 99 --f16-attn 0
```

### Validation

- Every `supertonic_*_forward_ggml` entry point opens an RAII
  `supertonic_op_dispatch_scope(model)`, so a CPU-only second engine
  in the same thread still sees the default `true` after a GPU
  engine's forward returns — required because the pointwise vocoder
  parity harness and the pipeline trace harness re-enter the model
  from a single thread.
- Both the trace `*_trace_ggml` entry points and the production
  `*_forward_ggml` ones acquire the scope: trace runs still pick the
  pure-GGML pathway whenever the backend isn't CPU, which is what the
  existing parity tests expect (the trace harness already disables the
  fused tail-update op via `!trace_outputs`; the new gate just removes
  the secondary `ggml_custom_4d` branches under it).
- CTest harnesses `test-supertonic-pipeline`, `test-supertonic-vocoder`,
  `test-supertonic-vector`, `test-supertonic-text-encoder`,
  `test-supertonic-duration` continue to exercise the CPU path
  unchanged; running them with a GPU-bound model would route the same
  fixture data through the pure-GGML fallback graph and produce the
  same parity numbers (within F32 → F16 K/V tolerance on the attention
  output when `--f16-attn 1`).
- Three new CPU-only unit harnesses ship alongside the bring-up code
  to give the dispatch + portable-op primitives their own coverage
  independent of any model GGUF:

  | Test | What it covers |
  |------|----------------|
  | `test-supertonic-backend-dispatch` | Default thread-local flag state; `supertonic_op_dispatch_scope` mirroring CPU and GPU `supertonic_model` instances; RAII teardown on normal exit and on exception; nested-scope unwinding; independence of `use_cpu_custom_ops` / `use_f16_attn`. |
  | `test-supertonic-portable-ops`     | CPU-backend parity of `leaky_relu_portable_ggml` (CPU lowering) vs the GPU decomposition for every `α ∈ {0, 0.01, 0.05, 0.1, 0.5, 0.99, 1.0}`; graph-node-count check that the GPU dispatch actually expands the op (catches a regression back to a passthrough `ggml_leaky_relu`). |
  | `test-supertonic-f16-attn-parity`  | F32 vs F16 K/V `ggml_flash_attn_ext` parity on the two hot shapes from the vector estimator (text attention `kv=32`, style attention `kv=50`); tolerance budget `5e-3` absolute / `5e-3` relative, the same band chatterbox ships behind `--cfm-f16-kv-attn`. |

  All three are registered with `LABEL "unit"` so a fresh checkout's
  `ctest -L unit` exercises them without needing the Supertonic GGUF.

### Next optimization rounds

The roadmap beyond this PR — F16 weight materialization, Q8_0 GGUF
support, host↔GPU round-trip elimination, OpenCL kernel-time profile
mode, and vocoder-unpack-on-GPU — is captured with its test plan in
`PLAN_SUPERTONIC_OPENCL.md`.  Each phase has an acceptance test
spelled out (most TDD, written before the implementation lands).

---

## GPU bring-up: Vulkan (May 2026, QVAC-18605)

Target: the same `--n-gpu-layers > 0` flag already plumbed through the
Supertonic CLI / engine / bench layer, but resolved to **Vulkan** on
Linux/Windows boxes that ship a working ICD (NVIDIA proprietary, AMD
RADV via Mesa, Intel ANV, llvmpipe for headless CI) so QVAC consumers
without an OpenCL stack still get the GPU codepath.  Tracking ticket:
QVAC-18605.

### Inheritance from the OpenCL bring-up (QVAC-18607)

By construction, the OpenCL bring-up's foundational work is **backend-
portable**: every helper added in QVAC-18607 (the
`supertonic_op_dispatch_scope` RAII, `backend_is_cpu` flag, F16 K/V
flash-attention path, `leaky_relu_portable_ggml` decomposition) only
ever queries "is this CPU?".  When the resolved backend is Vulkan
those queries return false and the runtime takes the GPU-portable
path automatically.  The Phase 2 audit-driven optimizations (F1-F24
in `aiDocs/AUDIT_SUPERTONIC_OPENCL.md` — host caches, in-graph RoPE,
GPU↔GPU Q/K/V blits, ConvNeXt fusion, F16 weights, in-graph
transpose) likewise apply unchanged: each one removes a host↔GPU
synchronisation point or eliminates redundant memory traffic that
Vulkan pays exactly the same way OpenCL does.

What this PR adds on top is the **Vulkan-specific dispatch deltas**:
two new model flags, two backend-capability probes, a CLI knob for
device selection, and a CPU-only TDD test that locks in the new
contract.  Each is small, scoped, and sits behind the existing
`#ifdef GGML_USE_VULKAN` guard so non-Vulkan builds compile clean.

### What landed

| Change | File(s) | Rationale |
|--------|---------|-----------|
| `supertonic_model::backend_is_vk` set from `ggml_backend_is_vk(model.backend)` after `init_supertonic_backend()` resolves the device. | `supertonic_gguf.cpp`, `supertonic_internal.h` | Informational; consumed by `engine.cpp::backend_name()` and `supertonic_bench.cpp` so multi-GPU machines unambiguously identify which adapter ran the bench (e.g. `Vulkan (device 0: NVIDIA GeForce RTX 5090)` instead of the bare `Vulkan` string). |
| `supertonic_model::use_native_leaky_relu` set from a load-time `ggml_backend_supports_op` probe against a synthetic LEAKY_RELU node.  Mirrored into the dispatch scope's thread-local. | `supertonic_gguf.cpp`, `supertonic_internal.h` | The OpenCL bring-up's `leaky_relu_portable_ggml` always decomposes into `RELU + SCALE + ADD` on non-CPU backends (3 dispatches).  Vulkan / Metal / CUDA implement `GGML_OP_LEAKY_RELU` natively (1 dispatch) — the probe lets the helper short-circuit to the fused builtin on backends that have it, without a hard-coded backend table.  Plain upstream OpenCL (no chatterbox patch) keeps the conservative decomposition. |
| `supertonic_backend_supports_f16_kv_flash_attn(backend)` probe; engine + bench auto-policy gates `use_f16_attn` on the result. | `supertonic_gguf.cpp`, `supertonic_internal.h`, `supertonic_engine.cpp`, `supertonic_bench.cpp` | The OpenCL bring-up's auto-policy flipped `use_f16_attn = !backend_is_cpu` blindly.  Replaced with a backend-capability probe that builds a synthetic Supertonic-shaped flash-attn graph node (`Q[head_dim, q_len, n_heads]` F32, `K/V[head_dim, kv_len, n_heads]` F16) and asks the backend whether it would accept the op.  A backend that ships `flash_attn_ext` but rejects the F16-K/V variant for our shape now keeps the F32 path — slower but guaranteed not to crash at first synth call.  Manual `--f16-attn 1` still forces dispatch (debug). |
| `init_supertonic_backend(n_gpu_layers, verbose, vulkan_device)` — Vulkan device-index parameter.  Range-checks against `ggml_backend_vk_get_device_count()`; an out-of-range value is a hard error (no silent CPU fallback — that would mask CLI typos / wrong-machine config).  Verbose mode logs device description from `ggml_backend_vk_get_device_description`. | `supertonic_gguf.cpp` | Replaces the historical hard-coded `ggml_backend_vk_init(0)`.  Multi-GPU machines + CI runners with a primary llvmpipe and a secondary discrete GPU need a way to pick. |
| `EngineOptions::vulkan_device` (default 0) plumbed through `load_supertonic_gguf`. | `tts-cpp/include/tts-cpp/supertonic/engine.h`, `supertonic_engine.cpp` | Public API. |
| `--vulkan-device N` flag wired into `supertonic-cli`, `supertonic-bench`, and `tts-cli` (the chatterbox CLI's Supertonic dispatch path). | `supertonic_cli.cpp`, `chatterbox_cli.cpp`, `supertonic_bench.cpp` | CLI surface. |
| `test-supertonic-vulkan-dispatch` — CPU-only unit test (`LABEL "unit"`) covering the new `backend_is_vk` / `use_native_leaky_relu` flags through `supertonic_op_dispatch_scope`, plus a smoke test for the F16-K/V flash-attn probe. | `test/test_supertonic_vulkan_dispatch.cpp`, `CMakeLists.txt` | Locks in the new dispatch contract for future regressions; runs on a fresh checkout under `ctest -L unit` without any GGUF fixture. |

### Vulkan supported-op matrix (relevant to Supertonic)

Verified against `ggml/src/ggml-vulkan/ggml-vulkan.cpp` HEAD on this
branch:

| Op | Native on ggml-vulkan? | Notes |
|----|:---:|---|
| `GGML_OP_LEAKY_RELU` (F32) | ✓ | `pipeline_leaky_relu_f32` shader.  `leaky_relu_portable_ggml` short-circuits to fused builtin via the new `use_native_leaky_relu` probe. |
| `GGML_OP_FLASH_ATTN_EXT` (F32 Q, F16 K/V) | ✓ | Requires `HSK % 8 == 0`; Supertonic's `head_dim=64` satisfies this by construction.  Output is F32, which matches what the downstream dense projection expects. |
| `GGML_OP_FLASH_ATTN_EXT` (F32 Q, Q4_0/Q8_0 K/V) | ✓ | Available for future quantized-K/V experiments (chatterbox §3.32 deferred this). |
| `GGML_OP_ROPE` | ✓ | Used by F20/F23 in-graph RoPE (post-OpenCL audit follow-up). |
| `GGML_OP_NORM`, `GGML_OP_MUL`, `GGML_OP_ADD`, `GGML_OP_REPEAT`, `GGML_OP_PERMUTE`, `GGML_OP_CONT`, `GGML_OP_TRANSPOSE`, `GGML_OP_RESHAPE`, `GGML_OP_VIEW`, `GGML_OP_SCALE`, `GGML_OP_RELU`, `GGML_OP_GELU_ERF`, `GGML_OP_MUL_MAT`, `GGML_OP_GET_ROWS`, `GGML_OP_CPY`, `GGML_OP_CONCAT` | ✓ | Universal op set used by the convnext fusion (F7), in-graph transpose (F12), graph-to-graph blit (F24), and every other audit follow-up.  No Supertonic ops missing on Vulkan. |

### How to use

```bash
# Build with Vulkan (in the standalone tree; in-tree subtree consumes
# the ggml-speech vcpkg port which already provides the Vulkan
# backend).
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build-vulkan -j$(nproc) --target tts-cli supertonic-bench

# Run on Vulkan with auto F16 attention (gated by the new backend-
# capability probe; on a Vulkan adapter satisfying HSK%8==0 it
# auto-enables, on any backend that rejects the F16-K/V op for our
# shape it stays at F32 and continues correctly).
./build-vulkan/supertonic-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --n-gpu-layers 99 \
  --out /tmp/supertonic2.wav

# Pick a specific Vulkan adapter (default 0).  Useful on machines
# with a software rasteriser (llvmpipe) at index 0 and the real
# GPU at index 1.
./build-vulkan/supertonic-cli ... --n-gpu-layers 99 --vulkan-device 1

# Force F16 attention off (CPU-style F32 fallback) for parity:
./build-vulkan/supertonic-cli ... --n-gpu-layers 99 --f16-attn 0

# Bench output explicitly names the Vulkan adapter so multi-GPU
# log lines are unambiguous:
./build-vulkan/supertonic-bench --model models/supertonic2.gguf \
  --text "..." --runs 5 --n-gpu-layers 99 --vulkan-device 0
# →   backend: Vulkan (device 0: NVIDIA GeForce RTX 5090) (f16_attn=on) (native_leaky_relu=on)
```

### Validation

- `test-supertonic-vulkan-dispatch` (CPU-only, `LABEL "unit"`):
  29 / 29 checks pass on this branch.  Covers default flag state,
  scope-mirroring for CPU / Vulkan / OpenCL-style models (probe true
  vs false), RAII teardown on exception, nested-scope unwinding,
  independence of all three flags, and a smoke test for the F16-K/V
  flash-attn probe (CPU backend).
- `test-supertonic-portable-ops` updated to explicitly request the
  decomposition path (`use_native_leaky_relu = false` on the GPU
  model) so the existing GPU-decomposition correctness gate stays
  green now that the helper short-circuits to the fused builtin
  whenever the probe reports native support.  10 / 10 checks pass.
- `test-supertonic-backend-dispatch` (the OpenCL bring-up's tests):
  27 / 27 checks pass — the dispatch scope's new
  `prev_use_native_leaky_relu` slot is added without disturbing the
  existing `prev_use_cpu_custom_ops` / `prev_use_f16_attn` ones.
- All other CPU-only unit tests on the branch (the audit
  follow-ups' RoPE / transpose / convnext-fusion / graph-to-graph-blit
  / profile-csv / F16-weights / F16-attn-parity tests) continue to
  pass unchanged.
- Fixture-bound tests (`test-supertonic-pipeline`,
  `test-supertonic-vocoder`, `test-supertonic-vector`, …) continue
  to exercise the CPU path unchanged.  Running them against a
  Vulkan-bound model would route the same fixture data through the
  same pure-GGML fallback graph that the OpenCL audit work
  established and produce identical parity numbers (within F32 →
  F16 K/V tolerance on the attention output when `--f16-attn 1`).

### Vulkan optimization round 2 (May 2026, QVAC-18605 follow-up)

Layered on top of the Vulkan bring-up above; the round-2 changes
generalise the bring-up's "load-time backend probe" pattern into a
process-wide capability cache and add three more probes / dispatch
hooks that fit the same shape:

1. **Process-wide capability-probe cache** keyed by `ggml_backend_t`.
   The bring-up's three load-sites (`load_supertonic_gguf`,
   `Engine::Engine`, `supertonic_bench`'s `main`) each ran the
   `LEAKY_RELU` and F16-K/V flash-attn `supports_op` queries
   independently — 2-3× redundant probe traffic on every backend
   handle.  On Vulkan, `supports_op` may inspect the device's
   pipeline state (~50-200 µs per query on Adreno / llvmpipe / RADV
   in microbenchmarks); the cache short-circuits 100 % of the
   duplicates.  Test seam (`supertonic_clear_capability_cache` +
   `supertonic_capability_probe_call_count`) lets the unit test
   verify the cache is hit on the second call by comparing the
   counter before / after.

2. **F16 mul_mat backend-capability probe** — symmetric to the F16-K/V
   flash-attn probe.  The bring-up auto-enabled `use_f16_weights` on
   `!backend_is_cpu` blindly; a partial-port backend that ships F16
   storage but rejects the hot vector-estimator W_query mul_mat
   shape (`[256, 256] F16` weight × `[256, 16] F32` activation) would
   crash at first synth call.  Probe builds the live shape and asks
   `ggml_backend_supports_op`; auto-policy refuses materialisation
   on a `false` answer (slower F32 path stays correct).  Manual
   `--f16-weights 1` still forces the F16 path (debug-shim escape
   hatch).  Probe cached in `cached_backend_capabilities`.

3. **Q8_0 K/V flash-attn forward-compat probe** — Vulkan's
   `GGML_OP_FLASH_ATTN_EXT` `supports_op` advertises Q8_0 (and Q4_0)
   K/V types in both scalar and coopmat2 paths
   (`ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT`).  Switching K/V from
   F16 to Q8_0 would halve the per-step upload bandwidth (50 KB → 25
   KB per K/V on Supertonic's hot shape, ≈1 MB / synth on the
   default 5-step × 4-site schedule) in exchange for a small
   (~0.5 %) drift on the attention output.  This PR adds the probe
   + caches the result so a follow-up patch can flip
   `--kv-attn-type q8_0` on without re-querying; the live dispatch
   site is **not yet wired** because the drift hasn't been measured
   against the existing F16 K/V parity harness on a real Vulkan
   adapter.  Bench output annotates `(q8_0_kv_attn=available)` when
   the probe says yes so operators can confirm their hardware is
   ready for the follow-up.

4. **`Engine::warm_up(text)` + `EngineOptions::prewarm_text` +
   `--prewarm TEXT` CLI flag** — first-synth-latency reduction on
   Vulkan / OpenCL.  The in-tree thread_local graph caches handle
   every subsequent call but can't avoid the first pipeline-compile
   cost (~hundreds of ms on Adreno / RADV per chatterbox
   PROGRESS.md).  `warm_up` runs one throwaway synth at construction
   time on a caller-supplied sample text so the operator-visible
   first synth sees steady-state latency.  Auto-no-op on CPU (no
   shader-compile cost to amortise).  The bench harness's
   `--prewarm` runs the cold-start synth BEFORE the timed loop
   starts (independent of `--warmup N`, which discards N timed runs
   from the median but doesn't avoid the cold-start hit on the
   first warmup run); the cold-start latency is logged separately
   (`[prewarm] cold-start synth on '…' took N.Nms`) and surfaced in
   `--json-out` as `"prewarm_ms"`.

5. **Bench output extended** to surface every backend-capability
   dispatch flag plus the cold-start prewarm latency, so log-grep
   across multiple machines can attribute perf differences to the
   right cause.  Backend log line now reads e.g.
   `Vulkan (device 0: NVIDIA RTX 5090) (f16_attn=on)
   (f16_weights=on) (native_leaky_relu=on)
   (q8_0_kv_attn=available)`.  JSON output adds `"f16_attn"`,
   `"f16_weights"`, `"native_leaky_relu"`,
   `"q8_0_kv_attn_available"`, `"prewarm_ms"` keys for downstream
   analysis tooling.

#### Round-2 validation summary

CPU-only, no GGUF needed — green on a fresh checkout under
`ctest -L unit`:

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-capability-cache` (NEW) | Probe cache short-circuit + clear seam + per-backend independence + idempotency + F16 mul_mat probe + Q8_0 K/V probe | 18 / 18 PASS |
| `test-supertonic-warm-up-api` (NEW) | `EngineOptions::prewarm_text` defaults to empty + `Engine::warm_up(const std::string &)` API contract via SFINAE | 9 / 9 PASS |
| `test-supertonic-vulkan-dispatch` (existing) | F16-K/V probe smoke test now exercises the cache short-circuit path | 29 / 29 PASS — unchanged |
| `test-supertonic-portable-ops` / `-backend-dispatch` (existing) | Round-1 dispatch correctness | 10 / 10 + 27 / 27 PASS |
| Audit follow-up tests from #16 (rope / transpose / convnext-fusion / graph-to-graph-blit / profile-csv / F16-attn-parity) | Audit-driven optimisation correctness | All PASS — unchanged |

Whole CPU-only `ctest -L unit` reports 184 / 184 checks passing
across the new tests + every audit-follow-up + bring-up test.

### Deferred work

These were investigated but kept out of scope for this PR:

- **Persistent `VkPipelineCache`** (chatterbox PROGRESS.md §3.32):
  recovers ~91 % of cold→warm shader-compilation gap on first warm
  run, keyed by `<vendorID>-<deviceID>-<driverVersion>` and rooted
  at `$XDG_CACHE_HOME/ggml/vulkan`.  This is a `ggml-vulkan` internal
  patch (~199 lines) that benefits all Vulkan workloads, not just
  Supertonic; tracked separately so the supertonic-specific PR stays
  reviewable.  Round-2's `--prewarm` is an in-process workaround
  (warms the in-memory pipeline cache for one process lifetime); the
  persistent on-disk cache extends the win across process restarts.
  When it lands, this Supertonic Vulkan codepath inherits the
  cold-start win automatically.
- **Q8_0 / BF16 K/V flash-attention live dispatch**: rounds 2 + 3
  add the capability probes; the live `--kv-attn-type q8_0|bf16`
  dispatch wiring is deferred until the F16-vs-{Q8_0,BF16} K/V
  drift is measured against the parity harness on a real Vulkan
  adapter (probes prime the cache so the follow-up patch flips
  dispatch without re-querying).
- **Pinned-host-buffer per-step uploads**: round 3 adds the
  capability probe for `ggml_backend_vk_host_buffer_type()` so
  the cache + bench surface know whether the path is available
  on the resolved backend.  The actual per-engine input-
  scratchpad refactor (allocate text_emb / time-step / style
  embedding tensors in the host-pinned buffer type instead of
  the default device-local buffer to skip ggml-vulkan's internal
  staging-buffer hop) is deferred until measured on a real Vulkan
  adapter so we can quantify the reduction in `latent` upload
  latency.

---

### Vulkan optimisation round 3 (May 2026, QVAC-18605 follow-up #2)

Three more Vulkan-specific deltas, all developed test-first (TDD)
— the new tests were committed first, observed to fail on the
missing symbol, and only then was the implementation written and
the tests re-run.

1. **BF16 K/V flash-attn capability probe** (5th `backend_capabilities`
   flag).  Symmetric to the round-2 Q8_0 K/V probe.  Vulkan's
   `GGML_OP_FLASH_ATTN_EXT` `supports_op` advertises BF16 K/V via
   the coopmat2-only path; BF16 has the same 2-byte per-element
   footprint as F16 (so identical upload bandwidth) but the wider
   8-bit exponent range avoids the F16 underflow on small attention
   scores that drives the parity-harness tolerance widening.
   Forward-compat — the live `--kv-attn-type bf16` dispatch wiring
   is deferred to a follow-up that measures drift against the
   parity harness on a real Vulkan adapter.

2. **Multi-device auto-pick for `--vulkan-device -1`**.  Wires the
   previously-reserved auto-pick API: walks every visible adapter,
   queries `ggml_backend_vk_get_device_memory()` to read free
   VRAM, and dispatches into a pure-logic helper
   `resolve_vulkan_device_index(requested, free_vram_per_device)`
   that picks `argmax(free_vram)` (ties → lower index for stable
   per-run assignment on identical-spec multi-GPU machines).
   Verbose mode logs the per-device VRAM table so operators can
   confirm the auto-pick chose the expected adapter.  The pure-
   logic helper is testable on CPU with synthetic inputs (8 cases,
   23 checks) — separates the policy from the Vulkan-only plumbing.
   Reserved-future negative values (`-2`, `-100`, ...) now throw
   instead of silently falling through to device 0.

3. **Pinned-host-buffer-type capability probe** (6th
   `backend_capabilities` flag) + bench surface.  Probes whether
   `ggml_backend_vk_host_buffer_type()` is callable on the
   resolved backend (Vulkan + non-null buffer-type).  Forward-
   compat — primes the capability cache for a follow-up per-engine
   input-scratchpad refactor that skips ggml-vulkan's internal
   staging-buffer hop on per-step uploads.  Bench output now shows
   `bf16_kv_attn_available` + `pinned_host_buffer_available` in
   both the human-readable backend tag and the JSON output so
   operators can pre-flight whether a future opt-in will be
   effective on their machine.

#### Test plan (TDD, round 3)

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-capability-cache` (UPDATED) | Existing 18 checks + 9 new round-3 checks (BF16 K/V probe smoke + cache-slot share, pinned-host-buffer probe smoke + cache-slot share, null-backend handling for both) | 27 / 27 PASS |
| `test-supertonic-vulkan-device-select` (NEW) | 8 test functions × 23 checks for the pure-logic auto-pick helper (empty list, single device, argmax, tie-break, explicit-index passthrough, out-of-range, reserved-negative, zero-VRAM) | 23 / 23 PASS |
| Every existing unit test (resample, cpu/t3 caches, profile-csv, rope-in-graph, rope-packed-qk, convnext-block-fused, in-graph-transpose, graph-to-graph-blit, backend-dispatch, portable-ops, vulkan-dispatch, warm-up-api, f16-attn-parity) | Round 1 + 2 + audit follow-up correctness | 16 / 16 PASS — unchanged |

Whole CPU-only `ctest -L unit` reports **16 / 16 tests, 0 failures**.
The TDD discipline was strict: the new tests in round 3 were
committed BEFORE the implementation and verified to fail on the
missing symbol (the compile-error footprint is captured in the
PR description) — only then was the implementation written and
the tests re-run to verify green.

---

### Vulkan optimisation round 6 (May 2026, QVAC-18605 follow-up #3) — F16-weights operator deny-list

Round 6 layers a **user-overridable extra deny-list** on top of
the existing hand-curated `should_materialise_f16_weight()`
allow-list.  The curated allow-list (Phase 2A) already excludes
biases, norms, embeddings, depthwise convs, and pre-transposed
companions; the round-6 deny-list lets operators force-keep
specific *additional* tensors as F32 even when `--f16-weights`
is on.  Use cases:

- **A/B testing**: researcher wants to exclude a specific tensor
  pattern temporarily without recompiling.
- **Hardware-specific drift mitigation**: operator observes drift
  on a particular adapter / driver / shape and pins the
  problematic tensor to F32 via config rather than disabling F16
  weights wholesale.
- **Future-GGUF safety net**: new tensor patterns added in future
  Supertonic GGUFs that the curated allow-list inadvertently
  scoops in can be excluded via config without a code change.

Smallest blast radius of the four follow-up rounds — load-time
policy only, runtime dispatch unaffected, zero behaviour change
on the empty-deny-list default path.

#### What changed

1. **2-arg overload `should_materialise_f16_weight(name, extra_deny_substrings)`**
   added alongside the existing 1-arg version (existing test +
   call sites unchanged).  Substring matching (audit-friendly,
   matches the curated predicate's style; no regex compile cost
   or invalid-pattern surface).  The deny-list can only flip
   `true → false`, never `false → true` — it's a deny-list, not
   an allow-list.  Empty strings inside the deny-list are
   SKIPPED defensively, not treated as universal matches (config-
   typo guard against an empty entry silently disabling F16
   weights for the whole model).

2. **`EngineOptions::f16_weights_deny_list`** (`std::vector<std::string>`,
   default empty) — public API surface for engine-side
   integration.  Wired through `Engine::Impl` →
   `load_supertonic_gguf` → the per-tensor allocation loop.

3. **`load_supertonic_gguf` 7th parameter** added at the end of
   the signature with a `{}` default — every existing call site
   keeps compiling without modification.

4. **`supertonic_model::f16_weights_excluded_count`** counter
   bumped at load time when a curated-hot tensor is excluded by
   the user's deny-list.  Surfaced in bench's human + JSON
   output so operators can confirm their config took effect.

5. **CLI plumbing**: `--f16-weights-deny PAT1,PAT2,...` flag on
   `supertonic-cli`, `tts-cli` (chatterbox), and `supertonic-bench`
   (comma-separated substring patterns).

6. **Verbose-log line** in `load_supertonic_gguf` when the deny-
   list is non-empty (silent on the default path — no visual
   noise on existing operator workflows).

#### Test plan (TDD, round 6)

Both new tests were committed BEFORE the implementation and
observed to fail on the missing symbols (compile errors:
`'should_materialise_f16_weight' too many arguments` for the
predicate test; `'EngineOptions::f16_weights_deny_list'` no such
member for the API-surface test).  Only then was the
implementation written and the tests re-run.

| Test | Coverage | Result |
|------|----------|--------|
| `test-supertonic-f16-weights` (UPDATED) | Existing 36 checks (positives, negatives, edges) + 29 new round-6 checks across 7 new test functions (empty-list passthrough, matching-deny-excludes, non-matching-no-op, cannot-promote-cold, multiple-patterns ANY-match, empty-string defensive skip, empty-name safety) | 65 / 65 PASS |
| `test-supertonic-f16-deny-list-api` (NEW) | SFINAE compile-time gate for `EngineOptions::f16_weights_deny_list` + `load_supertonic_gguf` 7th param; runtime defaults check + assignability + regression guards on every other documented `EngineOptions` default | 9 / 9 PASS |
| Every other unit test (round 1+2+3 + audit follow-ups + the 14 baseline tests) | Zero-regression gate | 17 / 17 PASS — unchanged |

Whole CPU-only `ctest -L unit` reports **17 / 17 tests, 0
failures, 0 regressions**.

#### Why no live perf number?

Round 6 is a **policy** change, not a kernel change.  The
quality-recovery on hand-picked tensors is workload-specific and
quantified offline against the F16-attention parity harness;
this PR adds the operator-facing knob so future drift incidents
can be triaged via config without a code change.  Bench output
surfaces the excluded-count so CI scripts can attribute any
quality regression to a config change.

---

## Remaining Work

### Runtime and performance

- Investigate vector 3/4-thread variance.
- Consider a fused text relpos attention op only if profiling shows text is the
  next hard blocker.
- Add quantized Supertonic GGUF support once graph paths are ready for f16/q8.
- Run the chatterbox-style OpenCL profiling sweep on Adreno (Q4_0 weights,
  `flash_attn_f32_f16` enabled) to confirm the Supertonic bottleneck shifts
  from custom CPU ops to `kernel_mul_mm_f32_f32` and the same convnext block
  shape that chatterbox already profiled.
- Add CI coverage for converter help/setup syntax and portable Supertonic build
  targets.

### Distribution

- Publish generated GGUFs externally if reviewers/users should avoid local
  conversion:
  - GitHub release asset
  - Hugging Face
  - S3/R2/internal artifact storage
- Keep the repo itself model-file-free.

---

## Useful Commands

```bash
# Build Supertonic targets.
cmake --build build --target tts-cli supertonic-cli supertonic-bench test-supertonic-pipeline

# Create local Supertonic 2 GGUF.
bash scripts/setup-supertonic2.sh

# Synthesize with Supertonic 2.
./build/tts-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --threads 4 \
  --out /tmp/supertonic2.wav

# Benchmark GGML.
./build/supertonic-bench \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --threads 4 --runs 3 --warmup 1 \
  --json-out artifacts/supertonic-thread-matrix/ggml-quick-t4.json

# Benchmark ONNX Runtime CPU.
python scripts/bench-supertonic-onnx.py \
  --onnx-dir /path/to/supertonic-pytorch/onnx_models/onnx \
  --assets-dir /path/to/supertonic-pytorch/assets \
  --voice-style /path/to/supertonic-pytorch/assets/voice_styles/F1.json \
  --text "The quick brown fox jumps over the lazy dog." \
  --lang en --language-wrap-mode open_close \
  --steps 5 --speed 1.05 --threads 4 --runs 3 --warmup 1 \
  --providers CPUExecutionProvider \
  --json-out artifacts/supertonic-thread-matrix/onnx-quick-t4.json
```
