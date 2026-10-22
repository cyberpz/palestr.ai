"""Palestra runner — local LLM bench orchestrator (stdlib only).

Per modello: avvia llama-server, aspetta /health, esegue i task,
salva i risultati (append per-task), spegne il server.

Uso:
  python runner.py --tasks smoke                 # tutti i modelli
  python runner.py --tasks smoke --only qwen3.8-27b
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE, "hf-cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE, "hf-cache", "datasets")
# Corsie parallele: lane -> file risultati proprio; gpus -> CUDA_VISIBLE_DEVICES
LANE = os.environ.get("PALESTRA_LANE", "main")
RESULTS = os.path.join(BASE, "results", "results-%s.json" % LANE)
LOGS = os.path.join(BASE, "logs")
GPUS = os.environ.get("PALESTRA_GPUS", "")
PORT = int(os.environ.get("PALESTRA_PORT", "0") or 0)
CTX_OVERRIDE = int(os.environ.get("PALESTRA_CTX", "0") or 0)


def http_json(url, payload=None, timeout=600):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def load_config():
    with open(os.path.join(BASE, "models.json"), encoding="utf-8") as f:
        return json.load(f)


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(rows):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    tmp = RESULTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    os.replace(tmp, RESULTS)
    push_dashboard(rows)


def push_dashboard(rows):
    cfg = load_config().get("dashboard")
    if not cfg:
        return
    try:
        req = urllib.request.Request(
            cfg["url"],
            data=json.dumps(rows).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "X-Palestra-Token": cfg["token"]},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print("   (dashboard push fallito: %s)" % e, flush=True)


def load_tasks(names):
    tasks = []
    for n in names:
        path = os.path.join(BASE, "tasks", n + ".py")
        spec = importlib.util.spec_from_file_location("task_" + n, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tasks.append((n, mod))
    return tasks


class Server:
    def __init__(self, cfg, model, port):
        self.cfg = cfg
        self.model = model
        self.port = port
        self.native_port = port + 1
        self.proc = None
        self.shim = None
        self.log_path = os.path.join(
            LOGS, "%s_%s.log" % (model["id"], int(time.time()))
        )

    def __enter__(self):
        args = [
            self.model.get("engine", self.cfg["engine"]),
            "-m", self.model["path"],
            "--host", "127.0.0.1",
            "--port", str(self.native_port),
            "-c", str(CTX_OVERRIDE or self.model.get("ctx", 8192)),
            "-ngl", "99",
            "--no-webui",
        ]
        if self.model.get("mmproj"):
            args += ["--mmproj", self.model["mmproj"]]
        if self.model.get("draft"):
            args += ["-md", self.model["draft"]]
        args += self.model.get("extra_args", [])
        os.makedirs(LOGS, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        env = os.environ.copy()
        if GPUS:
            env["CUDA_VISIBLE_DEVICES"] = GPUS
        self.proc = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "llama-server morto all'avvio (exit %s), log: %s"
                    % (self.proc.returncode, self.log_path)
                )
            try:
                st = http_json("http://127.0.0.1:%d/health" % self.native_port, timeout=5)
                if st.get("status") == "ok":
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("timeout health, log: %s" % self.log_path)
        # shim OpenAI-compat davanti al native
        self.shim = subprocess.Popen(
            [sys.executable, os.path.join(BASE, "shim.py"),
             str(self.port), str(self.native_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        time.sleep(2)
        return self

    def __exit__(self, *exc):
        for p in (self.shim, self.proc):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
        time.sleep(5)  # rilascio VRAM
        return False


def is_done(rows, model_id, task):
    for r in rows:
        if r["model"] == model_id and r["task"] == task:
            res = r.get("result", {})
            if res.get("error"):
                continue
            if task == "smoke":
                return True
            if res.get("score") is not None:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="smoke", help="nomi task separati da ,")
    ap.add_argument("--only", default=None, help="un solo modello per id")
    ap.add_argument("--force", action="store_true", help="rigenera gia' fatti")
    args = ap.parse_args()

    cfg = load_config()
    tasks = load_tasks(args.tasks.split(","))
    models = cfg["models"]
    if args.only:
        models = [m for m in models if m["id"] == args.only]
        if not models:
            sys.exit("modello %s non nel registro" % args.only)

    rows = load_results()
    port = PORT or cfg["port"]
    for model in models:
        print("== %s ==" % model["id"], flush=True)
        pending = [t for t in tasks if args.force or not is_done(rows, model["id"], t[0])]
        if not pending:
            print("   (gia' completo, salto)", flush=True)
            continue
        try:
            with Server(cfg, model, port) as srv:
                for tname, tmod in pending:
                    print("   task %s ..." % tname, flush=True)
                    try:
                        res = tmod.run("http://127.0.0.1:%d" % port, model)
                    except Exception as e:
                        print("   !! task %s fallito: %s" % (tname, e), flush=True)
                        res = {"error": str(e)}
                    rows.append({
                        "model": model["id"],
                        "task": tname,
                        "engine": cfg["engine_label"],
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "result": res,
                    })
                    save_results(rows)
        except Exception as e:
            print("!! %s FALLITO: %s" % (model["id"], e), flush=True)
            rows.append({
                "model": model["id"],
                "task": "runner",
                "engine": cfg["engine_label"],
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": {"error": str(e)},
            })
            save_results(rows)
    print("== palestra finita ==")


if __name__ == "__main__":
    main()
