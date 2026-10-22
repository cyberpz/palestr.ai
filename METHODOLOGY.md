# Methodology

## Pipeline

```
lm-evaluation-harness (local-completions)
        │  OpenAI-style POST /v1/completions
        ▼
harness/shim.py  ── translates to llama.cpp native POST /completion
        │            (forwards stop sequences, retries empty completions)
        ▼
llama-server (CUDA, tensor-split across GPUs when needed)
```

`runner.py` orchestrates per model: start `llama-server` → wait `/health` →
run each task module → append result → stop server. Crash of one task never
kills the campaign; failed tasks are recorded with the error and skipped.

## Tasks

### smoke
600-token throughput probe (`n_predict=600`, temp 0) plus four sanity probes
(arithmetic, syllogism, JSON extraction, register rewrite). A model that
scores 0 everywhere or emits garbage is flagged before burning hours on full
suites.

### MMLU — generative, full
57 tasks, 0-shot, all 14,042 questions (no subsampling). Prompt ends with
`Answer:`; generation stops at newline; the answer is extracted with
`\b([A-D])\b` on the **last** match of the completion.

Pitfalls found the hard way:
- Thinking models emit blank lines / reasoning before the letter. Stopping at
  the first newline or matching a bare `[A-D]` silently scores ~0–30%.
- Matching without word boundaries hits words like "Based" → false positives
  (~31% inflation).
- The shim retries a completion **without** the newline stop when the first
  attempt returns empty (some models emit `\n` as the first token).

### GSM8K — 5-shot, full
Standard 5-shot. `until` stop sequences per problem boundary prevent the
runaway-generation failure mode (models that otherwise generate to the token
cap; this alone changed throughput from 8.6 s/it to 1.6 s/it).
Reported: `exact_match,strict-match` (primary) and flexible.

### IFEval — full
`prompt_level_strict_acc` (primary) and instruction-level strict. Generative,
cap 1024 tokens.

## Scoring honesty

- No subsampling anywhere.
- One run per cell; we do not re-roll until a better number.
- Invalid runs (protocol bugs discovered later) are **removed**, not hidden:
  see git history.
- Quantization differs per model because VRAM differs; we state the quant in
  every table.

## Hardware notes

- Models >12 GB run tensor-split across both GPUs (`--tensor-split`).
- Lanes: `PALESTRA_LANE`, `PALESTRA_GPUS`, `PALESTRA_PORT` isolate parallel
  runs (own results file, own GPU set, own port).
- KV cache sizing matters: ctx 8192 default; 12288 for long few-shot prompts.
