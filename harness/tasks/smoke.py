"""Task smoke: throughput + risposte di sanity (nessun punteggio automatico).

Serve a validare la palestra: tok/s reali, latenza prompt, e 4 prompt
fissi (matematica, logica, estrazione, italiano) il cui output verra'
salvato grezzo per revisione. Il punteggio arriva con l'harness.
"""
import json
import time
import urllib.request

FIXED_PROMPT = ("Spiega in tre righe perche' il Cielo e' blu. "
                "Rispondi in italiano.")

SANITY = [
    {"id": "math", "q": "Quanto fa 17 per 23 meno 19? Mostra solo il risultato finale dopo il ragionamento."},
    {"id": "logic", "q": "Tutti i Blop sono Razz. Alcuni Razz sono Fizz. Alcuni Blop sono Fizz. Vero, falso o incerto?"},
    {"id": "extract", "q": 'Extract name and city as JSON: "John Doe lives in Rome and invoices 100 euro."'},
    {"id": "italian", "q": "Riscrivi in italiano formale: 'tipo, il server sta morto, rifatelo pls'."},
]


def _post(base, path, payload, timeout=600):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def run(base, model):
    out = {}
    # 1) throughput: /completion, n_predict fisso
    try:
        r = _post(base, "/completion", {
            "prompt": FIXED_PROMPT, "n_predict": 128, "temperature": 0,
        })
        t = r.get("timings", {})
        out["tok_per_s"] = round(t.get("predicted_per_second", 0), 1)
        out["prompt_tokens"] = t.get("prompt_n")
        out["n_predicted"] = t.get("predicted_n")
    except Exception as e:
        out["tok_per_s_error"] = str(e)
    # 2) sanity chat
    answers = {}
    for s in SANITY:
        try:
            t0 = time.time()
            r = _post(base, "/v1/chat/completions", {
                "model": model["id"],
                "messages": [{"role": "user", "content": s["q"]}],
                "max_tokens": 800, "temperature": 0,
            })
            ch = r["choices"][0]["message"]
            answers[s["id"]] = {
                "content": (ch.get("content") or "")[:1200],
                "reasoning_len": len(ch.get("reasoning_content") or ""),
                "finish": r["choices"][0].get("finish_reason"),
                "secs": round(time.time() - t0, 1),
            }
        except Exception as e:
            answers[s["id"]] = {"error": str(e)}
    out["sanity"] = answers
    return out
