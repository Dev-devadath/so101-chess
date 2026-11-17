"""Re-measure the LEADER gripper's open/closed range when its reading is stuck at 0 or 100.

Symptom: leader_check.py shows the gripper pinned at 0.0 (or 100.0) while you move it.
Cause: the raw encoder now sits outside the calibrated range_min..range_max (the handle
slipped on the motor shaft or the motor was knocked), so lerobot clamps it.

Prints the raw encoder live. Squeeze the leader gripper fully SHUT, then open it fully,
a few times, then Ctrl+C. The measured min/max replace range_min/range_max for the
gripper only (JSON backup kept as <id>.<timestamp>.json.bak, also written to the servo EEPROM).
Homing offset and every other joint are left untouched.

    .venv/bin/python chess/leader_gripper_range.py
"""

import shutil
import time

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from teach_squares import LEADER_ID, LEADER_PORT


def main():
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect(calibrate=False)
    cal = leader.bus.calibration["gripper"]
    print(f"calibrated range {cal.range_min}..{cal.range_max}. Shut the gripper fully, open it fully, repeat; Ctrl+C when done.")
    lo, hi = 4095, 0
    try:
        while True:
            raw = leader.bus.sync_read("Present_Position", normalize=False)["gripper"]
            lo, hi = min(lo, raw), max(hi, raw)
            print(f"\r   raw {raw:4d}   seen {lo}..{hi}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    try:
        if hi - lo < 300:
            print(f"only {hi - lo} ticks of travel seen: the raw value is not following the handle. "
                  "Check the gripper motor's cable and horn screw; nothing saved.")
            return
        if input(f"replace {cal.range_min}..{cal.range_max} with {lo}..{hi}? [y/N] ").strip().lower() != "y":
            print("nothing saved")
            return
        cal.range_min, cal.range_max = lo, hi
        leader.bus.write_calibration(leader.bus.calibration)  # EEPROM first, so a failed write leaves the JSON matching the servo
        shutil.copy(leader.calibration_fpath, leader.calibration_fpath.with_name(f"{LEADER_ID}.{time.strftime('%Y%m%d-%H%M%S')}.json.bak"))
        leader._save_calibration()
        print(f"saved to {leader.calibration_fpath}; run leader_check.py to confirm the gripper reads 0..100")
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
