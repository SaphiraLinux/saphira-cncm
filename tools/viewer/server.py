#!/usr/bin/env python3
"""CNCM Sudoku Viewer relay (localhost-only dev tool).

Serves tools/viewer/viewer.html and relays live level-by-level solve
events from `cncm sudoku-stream --step` as JSON over HTTP.

Each browser session spawns exactly one solver subprocess, paused
between levels: POST /api/step writes one stdin line, which releases
exactly one model level. The relay never invents steps; pacing only.

Binds 127.0.0.1 exclusively. No auth: localhost-only by construction.
Stdlib only.
"""
import argparse
import http.server
import json
import os
import queue
import subprocess
import sys
import threading
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
MAX_SESSIONS = 8


def spawn_solver(cncm, conf, weights, adir, prefix, seed):
    cmd = [cncm, "sudoku-stream", "--conf", conf, "--weights", weights,
           "--asset-dir", adir, "--prefix", prefix, "--seed", str(seed),
           "--step"]
    # NOTE: stderr is inherited (relay log), never merged: stdout must stay
    # pure NDJSON, one event per line.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    q = queue.Queue()

    def pump():
        try:
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(None)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return proc, q


def kill_session(sid):
    with SESSIONS_LOCK:
        s = SESSIONS.pop(sid, None)
    if s:
        try:
            s["proc"].kill()
        except Exception:
            pass


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ViewerRelay/1"

    def log_message(self, *a):
        sys.stderr.write("relay: %s\n" % (a[0] % a[1:] if len(a) > 1 else a[0]))

    def _send(self, code, body, ctype="application/json"):
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/" or self.path == "/viewer.html":
            with open(os.path.join(HERE, "viewer.html"), "rb") as f:
                page = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path.startswith("/api/step"):
            sid = self.path.split("sid=")[-1].split("&")[0] if "sid=" in self.path else ""
            with SESSIONS_LOCK:
                s = SESSIONS.get(sid)
            if not s:
                self._send(404, json.dumps({"type": "error", "msg": "no such session"}))
                return
            # Drain-first: return an already-produced event WITHOUT pacing
            # the solver. Only release one stdin line when the solver is
            # actually waiting (queue empty); otherwise queued newlines
            # would run the solve ahead of the client and orphan events.
            try:
                line = s["q"].get_nowait()
            except queue.Empty:
                line = None
            if line is None:
                try:
                    s["proc"].stdin.write("\n")
                    s["proc"].stdin.flush()
                except Exception:
                    try:
                        line = s["q"].get_nowait()
                    except queue.Empty:
                        self._send(200, json.dumps({"type": "error",
                                                    "msg": "solver exited"}))
                        return
                    self._send(200, line.strip() or '{"type":"error","msg":"empty"}')
                    return
                try:
                    line = s["q"].get(timeout=self.server.step_timeout)
                except queue.Empty:
                    self._send(200, json.dumps({"type": "error",
                                                "msg": "solver timeout"}))
                    return
                if line is None:
                    self._send(200, json.dumps({"type": "error",
                                                "msg": "solver exited"}))
                    return
            self._send(200, line.strip() or '{"type":"error","msg":"empty"}')
            return
        self._send(404, json.dumps({"type": "error", "msg": "not found"}))

    def do_POST(self):
        if self.path not in ("/api/begin", "/api/end"):
            self._send(404, json.dumps({"type": "error", "msg": "not found"}))
            return
        try:
            ln = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            ln = 0
        try:
            body = json.loads(self.rfile.read(ln).decode("utf-8") or "{}")
        except Exception:
            self._send(400, json.dumps({"type": "error", "msg": "bad json"}))
            return
        if self.path == "/api/end":
            kill_session(str(body.get("sid", "")))
            self._send(200, json.dumps({"ok": True}))
            return
        prefix = str(body.get("prefix", ""))
        seed = int(body.get("seed", 0) or 0)
        if not prefix:
            self._send(400, json.dumps({"type": "error", "msg": "need prefix"}))
            return
        cfg = self.server.cfg
        try:
            proc, q = spawn_solver(cfg["cncm"], cfg["conf"], cfg["weights"],
                                   cfg["adir"], prefix, seed)
        except Exception as e:
            self._send(500, json.dumps({"type": "error", "msg": "spawn: %s" % e}))
            return
        sid = uuid.uuid4().hex[:12]
        with SESSIONS_LOCK:
            while len(SESSIONS) >= MAX_SESSIONS:
                old, _ = next(iter(SESSIONS.items()))
                SESSIONS.pop(old, None)
            SESSIONS[sid] = {"proc": proc, "q": q}
        self._send(200, json.dumps({"ok": True, "sid": sid}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cncm", default="cncm")
    ap.add_argument("--conf", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port), Handler)
    srv.cfg = {"cncm": args.cncm, "conf": args.conf,
               "weights": args.weights, "adir": args.asset_dir}
    srv.step_timeout = 180
    srv.daemon_threads = True
    print("viewer relay on http://127.0.0.1:%d/ (Ctrl-C to stop)" % args.port,
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
