# so101-chess

An SO-101 arm that moves chess pieces on a physical board. Pieces are picked by
a learned ACT policy and everything around the pick is inverse kinematics:

```
park -> IK hover above FROM -> ACT policy: descend, grasp, lift -> IK carry to TO
     -> IK lower + guided release -> hover -> next move / park
```

Status (2026-11-06): the pick policy `policies/act_pick_v1` picks every piece type
on every taught square (10/10 in the last test). Placing is IK only. The game loop
(`chess/play.py`: board reading, python-chess, Stockfish) plays full games on the arm.

## Hardware

- SO-101 follower arm (6x Feetech STS3215) with red TPU V-groove pads glued on both jaws
- SO-101 leader arm, used only to record demos
- Overhead USB camera on a rigid stand ("overhead"), wrist camera on the gripper ("wrist")
- Mac (Apple silicon, policy runs on MPS) for recording and play; a Linux box with a GPU for training
- 3D-printed board on a mat; squares a1..h5 are reachable (see `chess/reach_map.py`)

## Layout

```
chess/                     all the code; run every script from the repo root
  arm.py                   follower connection, FK/IK (placo + URDF), ramped moves, park/unpark
  reach_map.py             jaw/tool models, IK for a point with the jaws pointing down, reachability map
  board.py                 overhead image -> top-down 8x8 warp (board_calib.json)
  cameras.py               open cameras by name (macOS reshuffles indices)
  teach_squares.py         teach each square's real tool position with the leader (square_points.json)
  touch_calib.py           board position in the arm frame from touched squares
  fit_offsets.py           per-joint lerobot->URDF angle offsets (joint_offset.json)
  calibrate_squares_cam.py per-square correction measured with the overhead camera
  calibrate_board.py       click the board corners on the overhead feed (board_calib.json)
  calibrate_nadir.py       camera nadir from how tall pieces lean
  align_top_cam.py         re-aim a camera to match a recorded frame
  hover_squares.py         IK hover above taught squares (the pose the policy starts from)
  record_pick.py           record pick demos: IK hover, then the leader does descend/grasp/lift
  run_pick.py              run the pick policy with IK around it (hover, carry, place); the Mover class play.py uses
  play.py                  play a game: camera reads your move, Stockfish replies, the arm plays it
  place_check.py           measure where a placed piece landed (overhead camera)
  board_state.py           empty / white / black per square from the overhead camera
  gripper_check.py         gripper reading when closed on a piece (sets the thresholds)
  leader_check.py          watch the leader for encoder jumps
  cam_view.py              live camera view with fps
  read_pose.py             torque off, print joints and FK (measures arm.OFFSET)
  arm_calib.json           board frame in the arm frame (touch_calib)
  joint_offset.json        lerobot -> URDF joint offsets (fit_offsets)
  square_points.json       taught tool position per square (teach_squares + calibrate_squares_cam)
  board_calib.json         overhead board corners (calibrate_board)
  board_ref.png            empty-board reference for board_state.py
  top_ref_calib.jpg        overhead reference frame for align_top_cam.py
  so101_urdf/              SO-101 URDF (new calibration) and the meshes it references
policies/act_pick_v1/      the pick policy (lerobot pretrained_model folder; weights not in git, see below)
calibration/               lerobot servo calibration for the follower and leader (copy to ~/.cache, see Setup)
scripts/train_gpu.sh       copy a dataset to the GPU box, train ACT there, pull checkpoints
scripts/chess_act_env.sh   ports and defaults for the scripts
docs/DESIGN.md             the overall plan (hybrid IK + learned skills, game loop)
docs/RECORDING_PLAN.md     how the demos were recorded (rules, blocks, coverage)
datasets/                  LeRobotDataset folders (not in git; see Datasets)
outputs/                   training outputs pulled from the box (not in git)
```

## Setup

```sh
uv venv --python 3.12 .venv && uv pip install -r requirements.txt --override overrides.txt
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
cp calibration/my_follower.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
cp calibration/my_leader.json   ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```

Serial ports are set in `chess/arm.py` (follower) and `chess/teach_squares.py` (leader);
`scripts/chess_act_env.sh` has the same two. The policy weights: put
`model.safetensors` into `policies/act_pick_v1/` (from the HF Hub once pushed, or from
the training box).

## Workflow

1. Calibrate (once per physical setup): `fit_offsets.py` / `touch_calib.py` for the arm,
   `teach_squares.py` then `calibrate_squares_cam.py` for the squares, `calibrate_board.py`
   for the overhead camera. Check with `reach_map.py` (dry run, no arm) and `hover_squares.py`.
2. Check the hardware: `cam_view.py wrist|top`, `gripper_check.py`, `leader_check.py`.
3. Record demos: `record_pick.py` as in `docs/RECORDING_PLAN.md`.
4. Train on the box: `DATASET=<name> STEPS=100000 sh scripts/train_gpu.sh start`, then
   `log` and `pull`. The result lands in `outputs/act_<name>`.
5. Play a move: `.venv/bin/python chess/run_pick.py d2 d4` (defaults to `policies/act_pick_v1`;
   `--policy outputs/act_<name> --ckpt 080000` tries a training checkpoint).
6. Measure placement: `place_check.py`.
7. Play a game (`brew install stockfish` once): `board_state.py --ref` on the empty board, set up
   the pieces, park the arm, then `.venv/bin/python chess/play.py [--skill 5] [--no-arm]`.
   Captured pieces go to a drop zone 1.5 squares past h1, off the board: keep it clear.
   Logic tests without hardware: `.venv/bin/python tests/test_play.py`.

## Datasets

All are LeRobotDataset v3, 30 fps, two cameras (wrist 640x480, overhead warped to a
480x480 top-down board view), 6-D state and action (5 joints + gripper, lerobot degrees).
Every episode starts at the IK hover above a piece and ends with it lifted.

| Dataset | Episodes | Frames | Jaws | Notes |
|---|---|---|---|---|
| `so101_pick3_clean` | 112 | 13526 | red TPU V-groove pads | Training set of `act_pick_v1`. `so101_pick3` minus 12 bad demos. |
| `so101_pick3` | 124 | 15326 | red TPU V-groove pads | Raw recording, all h264. Record new demos into this one. |
| `so101_pick2` | 96 | 14166 | bare jaws, yellow string wrap | Earlier round. Its policy picked but less reliably: grasp heights varied 8–52 mm. |

`so101_pick3_clean` was made with lerobot's `delete_episodes` (episodes 6, 18, 48, 50, 83,
89, 101, 102, 107, 110, 120, 121: head grabs, no lift, fingertips >4 mm into the mat). It
re-encodes the files it cuts as AV1 and copies the rest as h264, so never record into it:
record into `so101_pick3` and rebuild. The jaw pads change what the wrist camera sees, so
do not mix `so101_pick2` with pick3 for a policy that runs with the pads.

Load with the pyav backend (torchcodec needs a system FFmpeg):

```python
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset("local/so101_pick3_clean", root="datasets/so101_pick3_clean", video_backend="pyav")
```

## Pushing to the Hugging Face Hub

Push a dataset and the policy with:

```sh
.venv/bin/hf auth login
.venv/bin/python -c "from lerobot.datasets import LeRobotDataset as D; D('<user>/so101_pick3_clean', root='datasets/so101_pick3_clean').push_to_hub()"
.venv/bin/hf upload <user>/so101-chess-act-pick policies/act_pick_v1 .
```
