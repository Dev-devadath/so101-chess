# act_pick_v1

ACT pick skill: from the IK hover above a piece (jaws open, pointing down), descend,
grasp, lift 15-40 mm. Inputs: `observation.images.wrist`, `observation.images.top`
(640x480), `observation.state` (5 joints + gripper, lerobot degrees). Output: joint
targets, chunk 100, n_action_steps 100.

- Dataset: `so101_pick3_clean` (112 episodes, 13526 frames). Recorded 2026-11-05 with
  the red TPU pads, every demo the same way (see `docs/RECORDING_PLAN.md`): pawns on all
  39 squares + the hard squares again, tall pieces on all squares, knight/bishop on ranks 4-5.
- Training: lerobot ACT defaults (resnet18, dim 512, 4 enc / 1 dec layers, VAE, lr 1e-5,
  amp), batch 8, 100k steps, seed 1000, RTX 4000 SFF Ada. Loss 6.8 -> 0.025.
  This folder is checkpoint 100000 (`outputs/act_so101_pick3_clean/checkpoints/100000/pretrained_model`).
- Result 2026-11-06: 10/10 first-attempt picks across all piece types, incl. rank 5 and the
  a/h files, 3.3-4.1 s per pick. Grasps 12-19 mm above the mat. Known weak spot: h1 (a cable
  sagged into the wrist camera at that pose during tests).
- Run-time thresholds (chess/run_pick.py): gripper empty < 7, closed-on-piece < 30, open 40.
