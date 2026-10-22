"""Shim OpenAI<->llama.cpp per lm-eval: converte /v1/completions con
echo+logprobs verso il native /completion (che questo build non espone
in forma OpenAI). stdlib only.

Uso: python shim.py <public_port> <native_port>
"""
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PUB = int(sys.argv[1])
NAT = int(sys.argv[2])
NATIVE = "http://127.0.0.1:%d" % NAT


def _native(payload):
    req = urllib.request.Request(
        NATIVE + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode("utf-8"))


def _to_openai(nat, model):
    chunks = nat.get("completion_probabilities") or []
    tokens = [c.get("content", "") for c in chunks]
    tlogprobs = []
    top = []
    for c in chunks:
        cps = c.get("completion_probabilities") or [{}]
        tlogprobs.append(cps[0].get("probs"))
        top.append({t["token"]: t["logprob"] for t in (c.get("top_logprobs") or [])})
    return {
        "id": "palestra",
        "object": "text_completion",
        "model": model,
        "choices": [{
            "index": 0,
            "text": nat.get("content", ""),
            "finish_reason": "stop",
            "logprobs": {"tokens": tokens, "token_logprobs": tlogprobs,
                         "top_logprobs": top, "text_offset": []},
        }],
        "usage": {"prompt_tokens": nat.get("tokens", 0),
                  "completion_tokens": nat.get("tokens_predicted", 0),
                  "total_tokens": nat.get("tokens", 0)},
    }


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self, code, obj):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            with urllib.request.urlopen(NATIVE + self.path, timeout=30) as r:
                body = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._reply(502, {"error": str(e)})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._reply(400, {"error": "bad json"})
        if self.path == "/v1/completions":
            nat = {
                "prompt": payload.get("prompt", ""),
                "n_predict": max(1, int(payload.get("max_tokens") or 1)),
                "temperature": payload.get("temperature", 1.0),
                "cache_prompt": True,
            }
            if payload.get("echo"):
                nat["echo"] = True
            if payload.get("stop"):
                stop = payload["stop"]
                nat["stop"] = [stop] if isinstance(stop, str) else list(stop)
            if payload.get("logprobs"):
                nat["logprobs"] = int(payload["logprobs"])
                nat["n_probs"] = int(payload["logprobs"])
            try:
                out = _to_openai(_native(nat), payload.get("model", ""))
                if not out["choices"][0]["text"].strip() and "\n" in nat.get("stop", []):
                    # Modelli che mettono la risposta DOPO capolinea (35B-A3B: "\n\nB"):
                    # retry senza stop su \n (cache_prompt rende economico) e lstrip.
                    nat2 = dict(nat)
                    nat2["stop"] = [s for s in nat["stop"] if s != "\n"]
                    out2 = _to_openai(_native(nat2), payload.get("model", ""))
                    if out2["choices"][0]["text"].strip():
                        out2["choices"][0]["text"] = out2["choices"][0]["text"].lstrip()
                        out = out2
            except Exception as e:
                return self._reply(502, {"error": str(e)})
            return self._reply(200, out)
        # resto: pass-through (chat, native completion)
        path = self.path if self.path in ("/completion", "/v1/chat/completions") else self.path
        try:
            out = _native_passthrough(path, payload)
        except Exception as e:
            return self._reply(502, {"error": str(e)})
        self._reply(200, out)


def _native_passthrough(path, payload):
    req = urllib.request.Request(
        NATIVE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode("utf-8"))


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PUB), H).serve_forever()
