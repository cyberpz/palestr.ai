"""MMLU generativa (57 subtask, 0-shot, estrazione lettera)."""
import lm_eval


def run(base, model):
    res = lm_eval.simple_evaluate(
        model="local-completions",
        model_args="model=%s,tokenizer=%s,base_url=%s/v1/completions,max_gen_toks=64,max_length=8192" % (model["id"], model.get("tokenizer", model["id"]), base),
        tasks=["mmlu_generative"],
        num_fewshot=0,
        log_samples=False,
    )
    out = {"subtasks": {}}
    accs = []
    for task, r in res["results"].items():
        acc = r.get("exact_match,get_response")
        if acc is not None:
            out["subtasks"][task] = round(acc, 4)
            accs.append(acc)
    out["score"] = round(sum(accs) / len(accs), 4) if accs else None
    out["n_subtasks"] = len(accs)
    return out
