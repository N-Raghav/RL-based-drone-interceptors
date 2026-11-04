#!/usr/bin/env python3
"""Hardware adapter for the AS-Defend-Real-v0 defender: drone_gui telemetry in, rc out.

    drone_gui (UDP telemetry, 127.0.0.1:9100) ──► this loop @ 13.5 Hz ──► POST /api/<iface>/rc
                                                  macro.py: zones + 3x5 Hungarian slots (same file as the sim)
                                                  policy_runtime.py: trained shared policy + safety layer
                                                  (--policy <checkpoint.pt>; without it: scripted planner)

It never takes off or lands: use the GUI for that. It only sends sticks to defenders the
telemetry says are flying, and drone_gui's deadman zeroes the sticks 0.35 s after this loop
stops. Ctrl+C stops it.

Frames: the macro layer works in the simulator's world frame (X out of the tag-10 wall,
Y to the right when facing the wall... see room_map.py: X = z_map, Y = x_map, Z up), metres.
Each telemetry pose is the camera in ONE tag's frame; config/map.yaml places that tag in the
tag-10 frame. --shared-frame treats every tag as tag 10 (for drone_gui --sim, where each
simulated drone flies in its own world in front of its own tag).

    python run_defender.py --defenders SIM-3,SIM-4,SIM-5 --attackers SIM-1,SIM-2 --shared-frame --drive-attackers
    python run_defender.py --defenders TELLO-1,TELLO-2,TELLO-7 --attackers TELLO-3,TELLO-4 --site 2.5,0,1.25 \
        --policy shared_stage3_best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import socket
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent


def _find_map() -> Path:
    """The room's tag map: works from drone_deployment_infra/defender/, from <repo>/defender/, or
    with a map.yaml copied next to this file. --map overrides."""
    for c in (HERE / "map.yaml", HERE.parent / "organiser_base/drone_deployment_infra/config/map.yaml",
              HERE.parent / "config/map.yaml",
              HERE.parent / "drone_deployment_infra/config/map.yaml", HERE.parents[1] / "drone_deployment_infra/config/map.yaml"):
        if c.exists():
            return c
    return HERE.parent / "config/map.yaml"


# the planner ships in this folder (a byte-identical copy of the simulator's macro.py, checked by
# tests/test_policy_runtime.py), so this folder + a checkpoint is the whole deliverable
MACRO_PATH = HERE / "macro.py"
MAP_PATH = _find_map()  # the room's tag map (override: --map)

_spec = importlib.util.spec_from_file_location("defend_macro", MACRO_PATH)
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M
_spec.loader.exec_module(M)

sys.path.insert(0, str(HERE))
from policy_runtime import SharedPolicy, build_obs, safety  # noqa: E402

TAG10_HEIGHT = 1.22  # [m] tag 10 centre above the floor (same constant as the sim)
VIDEO_DELAY = 0.2  # [s] camera pipeline delay not included in telemetry "age"
LOST_AGE = 0.4  # [s] own fix older than this: stop translating, turn back to the tags
STALE_TELEMETRY = 0.5  # [s] no datagram this long: send nothing (GUI deadman zeroes sticks)


# -- frames ---------------------------------------------------------------------------------


class Frames:
    """Telemetry pose (camera in a tag's frame) -> world position and heading."""

    def __init__(self, shared: bool, map_path: Path = MAP_PATH):
        self.shared = shared
        self.tags = {}
        if not shared and not Path(map_path).exists():
            sys.exit(f"no tag map at {map_path}: pass --map <map.yaml> (written by cal_and_map 05_map.py)")
        if Path(map_path).exists():
            m = yaml.safe_load(Path(map_path).read_text())
            for tid, v in (m.get("tags") or {}).items():
                self.tags[int(tid)] = (np.array(v["R"], float), np.array(v["t"], float))

    def tag_pose(self, tag: int):
        if self.shared:
            return np.eye(3), np.zeros(3)
        return self.tags.get(int(tag))

    @staticmethod
    def to_world(p0: np.ndarray) -> np.ndarray:
        """tag-10 (map) frame -> sim world frame: X = z_map, Y = x_map, Z = y_map + tag10 height."""
        return np.array([p0[2], p0[0], p0[1] + TAG10_HEIGHT])

    def pose(self, pose: dict):
        """Returns (world position (3,), world forward unit vector (3,)) or None if the tag is unmapped."""
        tp = self.tag_pose(pose["tag"])
        if tp is None:
            return None
        R, t = tp
        x, y, z = pose["xyz"]  # telemetry: x right, y down, z into the wall (camera in front: z < 0)
        p_tag = np.array([x, -y, -z])  # map tag frame: x right, y up, z out of the face
        pos = self.to_world(R @ p_tag + t)
        psi = math.radians(pose.get("yaw", 0.0))  # 0 = facing the tag, + = turned toward +x
        f_tag = np.array([math.sin(psi), 0.0, -math.cos(psi)])
        f0 = R @ f_tag
        fwd = np.array([f0[2], f0[0], f0[1]])
        fwd[2] = 0.0
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        return pos, fwd


# -- per-drone tracking ---------------------------------------------------------------------


class Track:
    """Latest world fix, predicted forward by its age; velocity by finite difference."""

    def __init__(self, name: str):
        self.name = name
        self.hist: list[tuple[float, np.ndarray]] = []  # (fix time, world position)
        self.fwd = np.array([-1.0, 0.0, 0.0])
        self.age = 1e9
        self.tof = None
        self.flying = False

    def update(self, now: float, pos: np.ndarray, fwd: np.ndarray, age: float, tof):
        t_fix = now - age - VIDEO_DELAY
        if not self.hist or t_fix > self.hist[-1][0] + 1e-3:
            self.hist.append((t_fix, pos))
            self.hist = [h for h in self.hist if t_fix - h[0] <= 1.0]
        self.fwd = fwd
        self.age = age + VIDEO_DELAY
        self.tof = tof

    def velocity(self, window: float = 0.3) -> np.ndarray:
        if len(self.hist) < 2:
            return np.zeros(3)
        t1, p1 = self.hist[-1]
        older = [h for h in self.hist if t1 - h[0] >= window] or self.hist[:1]
        t0, p0 = older[-1]
        return (p1 - p0) / max(t1 - t0, 1e-3)

    def estimate(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.hist:
            return None, None
        t1, p1 = self.hist[-1]
        v = self.velocity()
        if now - t1 > LOST_AGE:
            v = np.zeros(3)  # lost: hold the last fix; never extrapolate a stale velocity
        p = p1 + v * min(now - t1, LOST_AGE)
        if self.tof is not None and 0.1 <= self.tof <= 4.0:
            p = p.copy()
            p[2] = self.tof  # height from the downward range sensor, not the tag pose
        return p, v


# -- external attacker feed -------------------------------------------------------------------


def read_attacker_feed(sock, tracks, alive, attackers, frame, now):
    """Drain the external attacker feed (UDP, one JSON object per datagram) into the tracks.

        {"attackers": {"<name>": {"p": [x, y, z], "age": 0.05, "alive": true}, ...}}

    <name> must match --attackers. p in metres, in --feed-frame. age (optional, s) is how old
    the measurement already is; alive (optional) false marks the attacker as down. A plain list
    [{"name": ..., "p": ...}, ...] under "attackers" is accepted too.
    """
    while True:
        try:
            data, _ = sock.recvfrom(65536)
        except BlockingIOError:
            return
        try:
            msg = json.loads(data)
        except ValueError:
            continue
        items = msg.get("attackers", {})
        if isinstance(items, list):
            items = {it.get("name") or it.get("id"): it for it in items}
        for name, it in items.items():
            name = str(name)
            if name not in attackers or "p" not in it:
                continue
            p = np.asarray(it["p"], dtype=float)
            if frame == "map":
                p = Frames.to_world(p)
            age = float(it.get("age", 0.0))
            tracks[name].update(now, p, np.array([-1.0, 0.0, 0.0]), age - VIDEO_DELAY, None)
            tracks[name].flying = True
            if it.get("alive") is False and alive.get(name, False):
                alive[name] = False
                print(f"[defender] feed: {name} is down", flush=True)


# -- rc ---------------------------------------------------------------------------------------


def to_rc(v_world: np.ndarray, fwd: np.ndarray, max_speed: float) -> np.ndarray:
    """World velocity (m/s) -> Tello rc a (right), b (forward), c (up) in -100..100."""
    right = np.array([fwd[1], -fwd[0], 0.0])
    body = np.array([v_world @ right, v_world @ fwd, v_world[2]])
    return np.clip(body / max_speed * 100.0, -100, 100)


def yaw_rc(fwd: np.ndarray, vel: np.ndarray | None = None, gain: float = 1.0, cap: float = 20.0) -> float:
    """Turn to face the direction of flight (the commanded world velocity): rc d, + = clockwise.

    Matches the sim's heading_mode="forward" (AS-Defend-Role-v0): below 0.2 m/s the drone keeps
    its heading. cap 20 (~20 deg/s): turning faster than ~0.2 rad/s blurs frames and costs fixes.
    Note the camera then sees the tag wall only while flying toward it.
    """
    if vel is None or math.hypot(vel[0], vel[1]) < 0.2:
        return 0.0
    heading = math.atan2(fwd[1], fwd[0])
    want = math.atan2(vel[1], vel[0])
    err = math.remainder(want - heading, 2 * math.pi)  # + means turn counter-clockwise
    return float(np.clip(-gain * math.degrees(err), -cap, cap))


class Gui:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.iface: dict[str, str] = {}

    def refresh(self):
        with urllib.request.urlopen(self.url + "/api/state", timeout=2) as r:
            state = json.load(r)
        self.iface = {d.get("drone") or d["name"]: d["iface"] for d in state["drones"]}
        self.iface.update({d["name"]: d["iface"] for d in state["drones"]})

    def rc(self, name: str, sticks):
        iface = self.iface.get(name)
        if iface is None:
            return
        body = json.dumps({"sticks": [int(round(s)) for s in sticks]}).encode()
        req = urllib.request.Request(f"{self.url}/api/{iface}/rc", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=0.05).read()
        except Exception:  # noqa: BLE001  (a missed packet is like rc loss; the next tick resends)
            pass


# -- main loop --------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defenders", required=True, help="comma-separated drone names, e.g. TELLO-1,TELLO-2,TELLO-7")
    ap.add_argument("--attackers", required=True, help="comma-separated drone names")
    ap.add_argument("--site", default="2.5,0,1.25", help="protected site centre in world frame X,Y,Z (m)")
    ap.add_argument("--port", type=int, default=9100, help="telemetry UDP port")
    ap.add_argument("--gui", default="http://127.0.0.1:8780")
    ap.add_argument("--hz", type=float, default=13.5)
    ap.add_argument("--rc-cap", type=float, default=40.0, help="|rc| cap (drone_gui clamps manual sticks at 40)")
    ap.add_argument("--rc-step", type=float, default=15.0, help="max rc change per tick (rc loss safety)")
    ap.add_argument("--max-speed", type=float, default=1.0, help="m/s at rc 100 (sim policy scale)")
    ap.add_argument("--site-radius", type=float, default=1.5)
    ap.add_argument("--shared-frame", action="store_true", help="treat every tag as tag 10 (drone_gui --sim)")
    ap.add_argument("--drive-attackers", action="store_true", help="test only: fly attackers straight at the site")
    ap.add_argument("--attacker-speed", type=float, default=0.3)
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = until Ctrl+C)")
    ap.add_argument("--log", default="outputs/defender_log.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, send no rc")
    ap.add_argument("--attacker-feed", choices=["gui", "udp"], default="gui",
                    help="where attacker positions come from: our drone_gui telemetry (tags), or an external UDP "
                         "JSON feed (the organisers'); see read_attacker_feed for the format")
    ap.add_argument("--feed-port", type=int, default=9200, help="UDP port of the external attacker feed")
    ap.add_argument("--feed-frame", choices=["map", "world"], default="map",
                    help="frame of the feed's positions: map = tag-10 frame (x right, y up, z out of the wall), "
                         "world = this script's frame (X out of the wall, Y right, Z up from the floor)")
    ap.add_argument("--map", default=str(MAP_PATH), help="room tag map (map.yaml)")
    ap.add_argument("--policy", default=None, help="trained shared-policy checkpoint (.pt); omit for the scripted planner")
    args = ap.parse_args()

    defenders = args.defenders.split(",")
    attackers = args.attackers.split(",")
    site = torch.tensor([[float(v) for v in args.site.split(",")]])
    cfg = M.MacroCfg(site_radius=args.site_radius, max_lin_speed=args.max_speed, solver="hungarian")
    state = M.MacroState(1, len(defenders), len(attackers), "cpu")
    policy = SharedPolicy.load(args.policy) if args.policy else None
    applied = torch.zeros(len(defenders), 4)  # last flown command, action units (a policy input)
    frames = Frames(args.shared_frame, Path(args.map))
    gui = Gui(args.gui)
    gui.refresh()
    need = defenders + (attackers if args.attacker_feed == "gui" or args.drive_attackers else [])
    missing = [n for n in need if n not in gui.iface]
    if missing:
        sys.exit(f"not in drone_gui: {missing}; known: {sorted(gui.iface)}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    feed = None
    if args.attacker_feed == "udp":
        feed = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        feed.bind(("0.0.0.0", args.feed_port))
        feed.setblocking(False)
        print(f"[defender] attacker feed: UDP :{args.feed_port}, {args.feed_frame} frame", flush=True)
    sock.setblocking(False)
    tracks = {n: Track(n) for n in defenders + attackers}
    alive = {n: True for n in attackers}
    last_rc = {n: np.zeros(4) for n in defenders + attackers}
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logf = open(args.log, "a")
    period = 1.0 / args.hz
    t_start = last_msg = time.time()
    last_print = 0.0
    print(f"[defender] defenders {defenders} attackers {attackers} site {args.site} "
          f"({'dry run' if args.dry_run else 'LIVE rc'})", flush=True)

    try:
        while True:
            tick = time.time()
            # drain telemetry, keep the newest datagram
            msg = None
            while True:
                try:
                    msg = json.loads(sock.recv(65536))
                except BlockingIOError:
                    break
            now = time.time()
            if feed is not None:
                read_attacker_feed(feed, tracks, alive, attackers, args.feed_frame, time.time())
            if msg is not None:
                last_msg = now
                for d in msg.get("drones", []):
                    tr = tracks.get(d["name"])
                    if tr is None or (feed is not None and d["name"] in attackers):
                        continue
                    tr.flying = bool(d.get("fly"))
                    if d.get("pose"):
                        res = frames.pose(d["pose"])
                        if res is not None:
                            tr.update(now, res[0], res[1], float(d["pose"].get("age", 0.0)), d.get("tof"))
            if now - last_msg > STALE_TELEMETRY:
                time.sleep(period)
                continue

            est = {n: tracks[n].estimate(now) for n in tracks}
            if any(est[n][0] is None for n in defenders + attackers):
                waiting = [n for n in tracks if est[n][0] is None]
                if now - last_print > 2:
                    print(f"[defender] waiting for a first tag fix from {waiting}", flush=True)
                    last_print = now
                time.sleep(period)
                continue

            # attackers inside the site count as breached (no longer tracked as threats)
            for n in attackers:
                p = est[n][0]
                if alive[n] and np.linalg.norm(p[:2] - site[0, :2].numpy()) < args.site_radius:
                    alive[n] = False
                    print(f"[defender] {n} reached the site", flush=True)

            dp = torch.tensor(np.stack([est[n][0] for n in defenders]), dtype=torch.float32)[None]
            dv = torch.tensor(np.stack([est[n][1] for n in defenders]), dtype=torch.float32)[None]
            apos = torch.tensor(np.stack([est[n][0] for n in attackers]), dtype=torch.float32)[None]
            avel = torch.tensor(np.stack([est[n][1] for n in attackers]), dtype=torch.float32)[None]
            al = torch.tensor([[alive[n] for n in attackers]])
            # a defender that has stopped flying (rammed an attacker, crashed, landed) is out:
            # the 3x5 assignment then covers both attackers with the drones still up
            def_up = torch.tensor([[tracks[n].flying for n in defenders]]) if not args.dry_run else None
            out = M.macro_step(dp, dv, apos, avel, al, site, state, cfg, def_alive=def_up)
            base = out["base"][0].numpy()  # (D, 4): world velocity / max_speed, yaw 0
            cmd = base
            if policy is not None:
                alive_d = def_up[0] if def_up is not None else None
                ages = torch.tensor([min(tracks[n].age, 5.0) for n in defenders], dtype=torch.float32)
                obs = build_obs(dp[0], dv[0], apos[0], avel[0], al[0], site[0], out, applied, ages, alive_d)
                act = policy.act(obs)
                cmd_t = safety(act, dp[0], alive_d)
                applied = cmd_t.clone()
                cmd = cmd_t.numpy()

            sent = {}
            for i, n in enumerate(defenders):
                tr = tracks[n]
                v = cmd[i, :3] * args.max_speed
                rc = np.zeros(4)
                if tr.age <= LOST_AGE:
                    rc[:3] = to_rc(v, tr.fwd, args.max_speed)
                rc[3] = yaw_rc(tr.fwd, v)
                sent[n] = rc
            if args.drive_attackers:
                for n in attackers:
                    tr = tracks[n]
                    to_site = site[0].numpy() - est[n][0]
                    to_site[2] = 0.0
                    v = args.attacker_speed * to_site / max(np.linalg.norm(to_site), 1e-6) if alive[n] else np.zeros(3)
                    rc = np.zeros(4)
                    rc[:3] = to_rc(v, tr.fwd, args.max_speed)
                    rc[3] = yaw_rc(tr.fwd, v)
                    sent[n] = rc
            for n, rc in sent.items():
                rc = np.clip(rc, last_rc[n] - args.rc_step, last_rc[n] + args.rc_step)
                rc = np.clip(rc, -args.rc_cap, args.rc_cap)
                last_rc[n] = rc
                if not args.dry_run and tracks[n].flying:
                    gui.rc(n, rc)

            slots = out["slot"][0].tolist()
            names = {-1: "post", 0: "prim A", 1: "prim B", 2: "sec A", 3: "sec B", 4: "goalie"}
            rec = {
                "t": round(now - t_start, 3),
                "def": {n: [round(float(x), 3) for x in est[n][0]] for n in defenders},
                "att": {n: [round(float(x), 3) for x in est[n][0]] for n in attackers},
                "slot": {n: names.get(s, s) for n, s in zip(defenders, slots)},
                "target": {n: [round(float(x), 3) for x in out["target"][0, i]] for i, n in enumerate(defenders)},
                "zone": out["zone"][0].tolist(),
                "pilot": "policy" if policy is not None else "scripted",
                "rc": {n: [int(round(x)) for x in last_rc[n]] for n in sent},
            }
            logf.write(json.dumps(rec) + "\n")
            if now - last_print > 1.0:
                gaps = []
                for n in defenders:
                    i = defenders.index(n)
                    gaps.append(f"{n}:{rec['slot'][n]} {np.linalg.norm(est[n][0] - out['target'][0, i].numpy()):.2f}m")
                att = " ".join(f"{n}:{np.linalg.norm(est[n][0][:2] - site[0, :2].numpy()):.1f}m" for n in attackers)
                print(f"[defender] t={rec['t']:6.1f} zones {rec['zone']} | {' | '.join(gaps)} | site dist {att}", flush=True)
                last_print = now
            if args.duration and now - t_start > args.duration:
                break
            time.sleep(max(0.0, period - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        logf.close()
        print("[defender] stopped (drone_gui deadman zeroes the sticks within 0.35 s)", flush=True)


if __name__ == "__main__":
    main()
