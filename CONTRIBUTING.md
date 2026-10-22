# Contributing

Contributions are welcome — especially new models, new rigs, and protocol fixes.

## Adding a model result

1. Run the suite with this harness (see README) on your rig.
2. Add `results/<model-id>.json` with your scores and run dates.
3. Add a row to the README table and state your hardware in the PR
   (GPU, VRAM, engine version). Scores from different rigs are welcome but
   must be labelled — we do not silently mix hardware in one table.

## Rules of the bench

- Full suites only. No subsampled "quick" numbers in the main table.
- One run per cell. No re-rolling until a nicer number.
- If you find a protocol bug, remove the invalidated results and say so in
  the PR. Honest history beats clean history.
- No personal data, hostnames, IPs, or tokens anywhere. `models.json` stays
  local; only `models.example.json` is committed.

## Protocol changes

Any change to prompts, stop sequences, or extraction regexes invalidates
previous scores for that task. Open an issue first so we can re-baseline.
