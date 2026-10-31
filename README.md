# Palestr.ai — Local LLM Bench Lab

Open-weight GGUF models benchmarked end-to-end on consumer hardware.
Full suites (no subsampling), a fully documented generative protocol, and
**every raw result committed to this repo**. No cherry-picking, no vibes.

## Results

Scores in %, `tok/s` from a 600-token throughput probe. `⏳` = run pending.

| Model | Quant | tok/s | MMLU-g | GSM8K | IFEval | Code* |
|---|---|---:|---:|---:|---:|---:|
| Mellum2-12B-A2.5B-Thinking | Q4_K_M | 120.5 | 70.0 | 81.1 | 27.2 | 69.9 |
| Spark-X2.5-4B | Q4_K_M | 55.6 | 67.2 | 81.0 | 19.6 | 62.4 |
| Spark-X2.6-4B | Q4_K_M | 53.6 | 65.7 | 60.6 | 16.8 | 64.0 |
| MiniCPM5-2B (heretic-abliterated) | Q8_0 | 100.0 | 56.2 | 73.6 | 34.4 | 72.5 |
| LFM2.5-24B-A2B | Q4_K_XL | **123.5** | 54.3 | 76.5 | 25.7 | 62.0 |
| Granite 4.2 3B | Q4_K_M | 101.3 | 51.6 | 75.3 | 26.1 | 56.4 |
| Granite 4.2 8B | Q4_K_M | 96.8 | 51.6 | 76.3 | 25.9 | 69.5 |
| Granite 4.2 30B | Q4_K_M | 17.5 | 69.0 | 89.3 | ✖ | 75.8 |
| Qwen3.8-27B | Q4_K_M | 45.5 | 35.8 | 80.8 | **50.8** | 85.7 |
| Qwen3.6-35B-A3B | IQ4_NL | 41.1 | ⏳ | 69.1 | 32.9 | 80.4 |
| Dory1.0-4B (NVIDIA) | Q5_K_M | — | ✖ | ✖ | ✖ | — |
| Ministral-3-8B-Instruct | Q4_K_M | 37.8 | **75.2** | 86.7 | 49.5 | 65.5 |
| flint 40B-A2.5B | Q3_K_XL | 63.7 | 66.8 | ✖ | ✖ | 76.0 |
| Nemotron Cascade-2 30B-A3B | IQ4_NL | — | 74.7 | 87.9 | 27.2 | **87.2** |
| Nemotron Bolt-74B-A9B-24GB | LiLiCorr | — | ✖ | ✖ | ✖ | 81.5 |
| LFM2.5-24B-A2B Uncensored-AD | Q4_K_M | 78.2 | 52.1 | 77.4 | 27.5 | — |
| Saluki-27B 1.0 (Underdog, abliterated) | IQ2-mix | 19.4 | 28.7 | **90.6** | 35.5 | 76.0 |

*Code = LiveCodeBench v6 pass@1 from official cards or independent
reproductions. — = not published.

Raw per-model JSON in [`results/`](results/). Scores are single runs,
greedy-ish decoding per task config (see [METHODOLOGY.md](METHODOLOGY.md)).

## What the benchmarks measure

**MMLU-g (generative MMLU)** — *general knowledge and broad reasoning.*
14,043 multiple-choice questions across 57 subjects (elementary math, US
history, computer science, law, medicine, ethics…). We run the generative
variant: the model reads the question and must **produce** the letter of the
correct answer, which is then extracted with a strict word-boundary regex.
This tests both knowledge recall and instruction-shaped output — a model can
know the answer and still fail to deliver it in the requested format.
Higher = better.

**GSM8K** — *grade-school math, i.e. multi-step reasoning.*
1,319 word problems that each require 2–8 arithmetic steps. The model must
show a chain of reasoning and land on the exact numeric answer. It is a
clean proxy for "can this model follow a logical procedure without
derailing", and it separates reasoning-tuned models from pure knowledge
models. Higher = better.

**IFEval** — *instruction following, the reliability benchmark.*
541 instructions with a programmatically checkable constraint ("answer in
exactly 3 bullet points", "include the word X", "respond in under 50
words"…). No knowledge involved: the score is the fraction of constraints
the output actually obeys (prompt-level, strict). This is the closest thing
to "will this model do what my automation asks" — a low IFEval with a high
MMLU means a knowledgeable model you cannot steer. Higher = better.

**tok/s** — raw generation throughput on a fixed smoke probe. On consumer
VRAM, speed is a first-class citizen: a 4× faster model is often the right
answer even at a small quality cost.

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
