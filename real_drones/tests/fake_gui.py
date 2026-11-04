"""A stand-in for drone_gui, for end-to-end tests of run_defender.py without drones.

Serves GET /api/state and POST /api/<iface>/rc on 127.0.0.1:<http_port>, integrates 5 point
drones from the sticks it receives (defenders) or flies them at the site (attackers), and
sends telemetry datagrams in drone_gui's wire format (drone_telemetry/README.md) to
127.0.0.1:<udp_port>. Poses are reported as the camera in tag 10's frame, like the real GUI.

    python fake_gui.py --defenders D1,D2,D3 --attackers A1,A2 --seconds 20 --out sticks.json
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TAG10_HEIGHT = 1.22


class Drone:
    def __init__(self, name, iface, pos, heading, attacker=False):
        self.name, self.iface, self.attacker = name, iface, attacker
        self.p = list(pos)  # world: X out of the tag wall, Y right, Z up
        self.h = heading  # world yaw, rad (0 = +X, pi = facing the wall)
        self.sticks = (0, 0, 0, 0)
        self.sticks_t = 0.0
        self.n_rc = 0

    def pose(self):
        # world -> camera in tag-10 frame: x right (=Y), y down (=-(Z-1.22)), z into the wall (=-X)
        psi = math.degrees(math.atan2(math.sin(self.h), -math.cos(self.h)))
        return {"tag": 10, "xyz": [round(self.p[1], 3), round(-(self.p[2] - TAG10_HEIGHT), 3), round(-self.p[0], 3)],
                "yaw": round(psi, 2), "age": 0.05}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defenders", default="D1,D2,D3")
    ap.add_argument("--attackers", default="A1,A2")
    ap.add_argument("--http-port", type=int, default=8780)
    ap.add_argument("--udp-port", type=int, default=9100)
    ap.add_argument("--site", default="2.5,0,1.25")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--max-speed", type=float, default=1.0, help="m/s at rc 100")
    ap.add_argument("--out", default="fake_gui_result.json")
    a = ap.parse_args()
    site = [float(v) for v in a.site.split(",")]
    D = a.defenders.split(",")
    A = a.attackers.split(",")
    drones = {}
    for i, n in enumerate(D):
        ang = 2 * math.pi * i / len(D)
        drones[n] = Drone(n, f"wlx_d{i}", [site[0] + 1.8 * math.cos(ang), site[1] + 1.8 * math.sin(ang), 1.2], math.pi)
    for j, n in enumerate(A):
        drones[n] = Drone(n, f"wlx_a{j}", [site[0] + 4.5, site[1] + (-1.5 if j == 0 else 1.8), 1.4], math.pi, attacker=True)
    by_iface = {d.iface: d for d in drones.values()}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/api/state":
                self._json({"drones": [{"name": d.name, "iface": d.iface, "drone": d.name} for d in drones.values()]})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "api" and parts[2] == "rc" and parts[1] in by_iface:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                st = tuple(max(-40, min(40, int(v))) for v in body.get("sticks", (0, 0, 0, 0)))  # drone_gui's cap
                with lock:
                    d = by_iface[parts[1]]
                    d.sticks, d.sticks_t = st, time.time()
                    d.n_rc += 1
                self._json({"ok": True})
            else:
                self._json({"error": "bad request"}, 400)

    srv = ThreadingHTTPServer(("127.0.0.1", a.http_port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt, t0, seq = 1 / 20, time.time(), 0
    breached = set()
    min_gap = {n: 1e9 for n in A}
    while time.time() - t0 < a.seconds:
        now = time.time()
        with lock:
            for d in drones.values():
                if d.attacker:
                    if d.name in breached:
                        continue
                    dx, dy = site[0] - d.p[0], site[1] - d.p[1]
                    r = math.hypot(dx, dy)
                    if r < 1.5:
                        breached.add(d.name)
                        continue
                    sp = 0.45 if now - t0 > 2 else 0.0  # attackers launch after 2 s
                    d.p[0] += sp * dx / r * dt
                    d.p[1] += sp * dy / r * dt
                    continue
                a_, b_, c_, y_ = d.sticks if now - d.sticks_t < 0.35 else (0, 0, 0, 0)  # deadman
                fwd = (math.cos(d.h), math.sin(d.h))
                right = (fwd[1], -fwd[0])
                s = a.max_speed / 100.0
                d.p[0] += (a_ * right[0] + b_ * fwd[0]) * s * dt
                d.p[1] += (a_ * right[1] + b_ * fwd[1]) * s * dt
                d.p[2] += c_ * s * dt
                d.h -= math.radians(y_ * 1.0) * dt  # rc d +: clockwise, ~1 deg/s per unit
            for n in A:
                for m in D:
                    g = math.dist(drones[n].p, drones[m].p)
                    min_gap[n] = min(min_gap[n], g)
            msg = {"v": 1, "src": "sim", "seq": seq, "t": now, "hz": 20.0, "drones": [
                {"name": d.name, "slot": i + 1, "team": "attack" if d.attacker else "defense", "link": True,
                 "fly": True, "mode": "manual", "bat": 80, "h": d.p[2], "tof": round(d.p[2], 3), "vel": [0, 0, 0],
                 "pose": d.pose()} for i, d in enumerate(drones.values())]}
        sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), ("127.0.0.1", a.udp_port))
        seq += 1
        time.sleep(dt)
    srv.shutdown()
    with lock:
        res = {"rc_posts": {d.name: d.n_rc for d in drones.values()},
               "final": {d.name: [round(v, 2) for v in d.p] for d in drones.values()},
               "breached": sorted(breached), "closest_defender_m": {k: round(v, 2) for k, v in min_gap.items()}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
