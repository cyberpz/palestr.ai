"""GSM8K full (5-shot, generative, stop at next problem)."""
import os

import lm_eval

YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yaml")


def run(base, model):
    res = lm_eval.simple_evaluate(
        model="local-completions",
        model_args="model=%s,tokenizer=%s,base_url=%s/v1/completions,max_gen_toks=1024,max_length=8192" % (model["id"], model.get("tokenizer", model["id"]), base),
        tasks=[os.path.join(YAML, "gsm8k_domus.yaml")],
        num_fewshot=5,
        log_samples=False,
    )
    r = res["results"].get("gsm8k", {})
    return {
        "score": r.get("exact_match,strict-match"),
        "flexible": r.get("exact_match,flexible-extract"),
    }
