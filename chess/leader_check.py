"""Watch the leader arm's readings for jumps. Pick it up and move it like a demo for 20 s.

Prints every tick where a joint changes by more than JUMP_DEG, with the raw encoder
ticks, then the min/max of every joint. Nothing is sent to the follower.

Usage:  .venv/bin/python chess/leader_check.py [seconds]
"""

import sys
import time

import numpy as np

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from arm import JOINTS
from teach_squares import LEADER_ID, LEADER_PORT

JUMP_DEG = 30.0
NAMES = JOINTS + ["gripper"]


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect()
    print(f"move the leader like a demo for {secs:.0f} s...")
    prev, lo, hi, jumps, n = None, None, None, 0, 0
    t_end = time.perf_counter() + secs
    try:
        while time.perf_counter() < t_end:
            a = leader.get_action()
            q = np.array([a[f"{j}.pos"] for j in NAMES])
            lo = q if lo is None else np.minimum(lo, q)
            hi = q if hi is None else np.maximum(hi, q)
            if prev is not None and np.abs(q - prev).max() > JUMP_DEG:
                jumps += 1
                raw = leader.bus.sync_read("Present_Position", normalize=False)
                j = NAMES[int(np.abs(q - prev).argmax())]
                print(f"JUMP {j}: {prev[NAMES.index(j)]:.0f} -> {q[NAMES.index(j)]:.0f} deg   raw ticks now {raw}")
            prev = q
            n += 1
            time.sleep(0.03)
    finally:
        leader.disconnect()
    print(f"{n} reads, {jumps} jumps")
    for j, a, b in zip(NAMES, lo, hi):
        print(f"   {j:14s} {a:7.1f} .. {b:7.1f}")


if __name__ == "__main__":
    main()
