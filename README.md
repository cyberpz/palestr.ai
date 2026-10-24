# Palestr.ai — Local LLM Bench Lab

Open-weight GGUF models benchmarked end-to-end on consumer hardware.
Full suites (no subsampling), a fully documented generative protocol, and
**every raw result committed to this repo**. No cherry-picking, no vibes.

## Results

Scores in %, `tok/s` from a 600-token throughput probe. `⏳` = run pending.

| Model | Quant | MMLU-g | GSM8K | IFEval | tok/s |
|---|---|---:|---:|---:|---:|
| Mellum2-12B-A2.5B-Thinking | Q4_K_M | **70.0** | **81.1** | 27.2 | 120.5 |
| Spark-X2.5-4B | Q4_K_M | 67.2 | **81.0** | 19.6 | 55.6 |
| Spark-X2.6-4B | Q4_K_M | 65.7 | 60.6 | 16.8 | 53.6 |
| MiniCPM5-2B (heretic-abliterated) | Q8_0 | 56.2 | 73.6 | 34.4 | 100.0 |
| LFM2.5-24B-A2B | Q4_K_XL | 54.3 | 76.5 | 25.7 | **123.5** |
| Granite 4.2 3B | Q4_K_M | 51.6 | 75.3 | 26.1 | 101.3 |
| Granite 4.2 8B | Q4_K_M | 51.6 | 76.3 | 25.9 | 96.8 |
| Granite 4.2 30B | Q4_K_M | 69.0 | 89.3 | ⏳ | 17.5 |
| Qwen3.8-27B | Q4_K_M | ⏳ | 80.8 | **50.8** | 45.5 |
| Qwen3.6-35B-A3B | IQ4_NL | ⏳ | 69.1 | 32.9 | 41.1 |
| Dory1.0-4B (NVIDIA) | Q5_K_M | sched | sched | sched | — |
| Ministral-3-8B-Instruct | Q4_K_M | sched | sched | sched | — |
| flint 40B-A2.5B | Q3_K_XL | sched | sched | sched | — |
| Nemotron Cascade-2 30B-A3B | IQ4_NL | sched | sched | sched | — |
| Nemotron Bolt-74B-A9B-24GB | LiLiCorr | sched | sched | sched | — |

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
