# Palestr.ai — Local LLM Bench Lab

Open-weight GGUF models benchmarked end-to-end on consumer hardware.
Full suites (no subsampling), a fully documented generative protocol, and
**every raw result committed to this repo**. No cherry-picking, no vibes.

## Results

Scores in %, `tok/s` from a 600-token throughput probe. `⏳` = run pending.

| Model | Quant | MMLU-g | GSM8K | IFEval | tok/s |
|---|---|---:|---:|---:|---:|
| Spark-X2.5-4B | Q4_K_M | **67.2** | **81.0** | 19.6 | 55.6 |
| MiniCPM5-2B (heretic-abliterated) | Q8_0 | 56.2 | 73.6 | 34.4 | 100.0 |
| LFM2.5-24B-A2B | Q4_K_XL | 54.3 | ⏳ | ⏳ | **123.5** |
| Qwen3.8-27B | Q4_K_M | ⏳ | 80.8 | **50.8** | 45.5 |
| Qwen3.6-35B-A3B | IQ4_NL | ⏳ | 69.1 | 32.9 | 41.1 |
| Mellum2-12B-A2.5B-Thinking | Q4_K_M | ⏳ | ⏳ | ⏳ | ⏳ |

Raw per-model JSON in [`results/`](results/). Scores are single runs,
greedy-ish decoding per task config (see [METHODOLOGY.md](METHODOLOGY.md)).

## Test rig

| | |
|---|---|
| GPU | 2× RTX 3060 12 GB (tensor-split for >12 GB models) |
| OS | Windows 11 |
| Engine | llama.cpp (CUDA) via `llama-server` |
| Harness | `lm-evaluation-harness` 0.4.13, `local-completions` backend |
| Glue | custom OpenAI↔native shim (`harness/shim.py`) |
| Suite | MMLU generative (57 tasks, 0-shot, full) · GSM8K (5-shot, full) · IFEval (prompt-strict, full) · smoke (throughput + sanity) |

## Why generative

llama.cpp removed `echo` support in 2026 builds, which kills loglikelihood
scoring through any HTTP server path. Instead of giving up we rebuilt the
scoring protocol around **generation + answer extraction**: strict stop
sequences, per-task extraction regexes, and a shim that restores `stop`
forwarding to the native endpoint. Details and pitfalls:
[METHODOLOGY.md](METHODOLOGY.md).

## Reproduce

```bash
# Windows or Linux, Python 3.11+, CUDA llama-server
pip install lm-eval==0.4.13 langdetect
cp harness/models.example.json harness/models.json   # edit paths
python harness/runner.py --tasks smoke --only <model-id>
python harness/runner.py --tasks smoke,mmlu,gsm8k,ifeval
```

Results land in `results/results-<lane>.json`. Multiple lanes (lanes =
`PALESTRA_LANE`/`PALESTRA_GPUS`/`PALESTRA_PORT` env) let you run small models
in parallel on separate GPUs.

## Honest limitations

- Single run per cell, no confidence intervals.
- MMLU-g (generative) is not directly comparable to loglikelihood MMLU numbers
  quoted elsewhere; expect generative to score lower on small models.
- Thinking models need the answer-extraction protocol in METHODOLOGY.md;
  naive matchers silently produce ~0 scores.
- GGUF quants differ per model (best fit for VRAM), so cross-model score
  deltas include quant effects.

## License

Code: MIT. Data (`results/`): CC BY 4.0. See `LICENSE` and `CONTRIBUTING.md`.
