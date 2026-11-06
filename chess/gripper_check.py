"""Print the gripper reading after closing on whatever is between the jaws.

Arm stays where it is (park it first, or hold a piece in the jaws by hand).
Each round: jaws open to 40, Enter, jaws close, reading printed.

Usage:  .venv/bin/python chess/gripper_check.py
"""

import time

from arm import connect, read_gripper, read_joints, send_joints

OPEN, CLOSE = 40.0, 0.0


def main():
    robot = connect()
    try:
        while True:
            send_joints(robot, read_joints(robot), gripper=OPEN)
            time.sleep(0.8)
            if input("jaws open. Put a piece in (or nothing), Enter to close, q to quit... ").strip() == "q":
                break
            send_joints(robot, read_joints(robot), gripper=CLOSE)
            time.sleep(1.0)
            print(f"   closed: gripper reads {read_gripper(robot):.1f}")
    finally:
        send_joints(robot, read_joints(robot), gripper=OPEN)
        time.sleep(0.5)
        robot.disconnect()


if __name__ == "__main__":
    main()
