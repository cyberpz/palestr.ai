"""HellaSwag full (0-shot, loglikelihood)."""
import lm_eval


def run(base, model):
    res = lm_eval.simple_evaluate(
        model="local-completions",
        model_args="model=%s,tokenizer=%s,base_url=%s/v1/completions,context_length=8192" % (model["id"], model.get("tokenizer", model["id"]), base),
        tasks=["hellaswag"],
        num_fewshot=0,
        log_samples=False,
    )
    r = res["results"].get("hellaswag", {})
    return {"score": r.get("acc,none"), "n": r.get("n,none")}
