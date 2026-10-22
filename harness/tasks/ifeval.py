"""IFEval full (prompt-strict, generative, cap 1024)."""
import os

import lm_eval

YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yaml")


def run(base, model):
    res = lm_eval.simple_evaluate(
        model="local-completions",
        model_args="model=%s,tokenizer=%s,base_url=%s/v1/completions,max_length=8192" % (model["id"], model.get("tokenizer", model["id"]), base),
        tasks=[os.path.join(YAML, "ifeval_domus.yaml")],
        num_fewshot=0,
        log_samples=False,
    )
    r = res["results"].get("ifeval", {})
    return {
        "score": r.get("prompt_level_strict_acc,none"),
        "inst_strict": r.get("inst_level_strict_acc,none"),
    }
