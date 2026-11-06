# Recording plan: so101_pick3 (red TPU pads on)

Why a new dataset: the 96 episodes in so101_pick2 were recorded with the bare
(yellow, string-wrapped) jaws and grasp heights all over the place (8–52 mm above
the board). The policy learned that spread. This set is recorded with the pads on
and with every demo done the same way. Train on so101_pick3 alone first.

## Rules for every episode

1. Leader gripper open to about 40 before Enter (the script waits until it is 35–45).
2. Go straight down. No sideways searching: if the piece is not between the jaws

   at the hover, press `r`, fix the piece on the square, redo.
3. Watch the live `tip` readout. Close the jaws when it reads:
   - pawn: 10–15 mm
   - knight, bishop, rook, king, queen: at the widest part of the body,
   roughly 15–25 mm (do not grab the head or the neck)
   - never below 5 mm (that is the fingertips in the mat)
4. Close the leader gripper fully and hold it closed.
5. Lift straight up until the tip reads 40–50 mm, hold still, Enter.
6. If the piece slipped, tilted or was grabbed by the head: `r` and redo. Only

   clean episodes go in.

Keep the one or two neighbouring pieces placed around the target as usual.

## Blocks (run in this order, ~116 episodes)

Block 1: pawn only, every square once (39). Builds the base skill on the hardest piece.

```
.venv/bin/python chess/record_pick.py --dataset so101_pick3 --episodes 39 --swap-every 0 g1 d5 f2 e3 a3 d2 h4 g3 b2 e2 h5 a1 f4 h2 g5 d3 h3 a5 e4 b4 c3 b5 a2 d4 f5 g2 c1 b1 c5 b3 g4 e5 a4 h1 f1 e1 c4 c2 f3
```

Block 2: pawn only, the hard squares again: ranks 4–5 plus the a and h files near the base (22).

```
.venv/bin/python chess/record_pick.py --dataset so101_pick3 --resume --episodes 22 --swap-every 0 c5 h3 e5 c4 g5 e4 h2 h4 h5 f5 a4 b5 b4 a1 a5 h1 d5 g4 a3 d4 f4 a2
```

Block 3: tall pieces, every square once (39). Swap every 8: rook, bishop, queen, king, knight, then repeat.

```
.venv/bin/python chess/record_pick.py --dataset so101_pick3 --resume --episodes 39 --swap-every 8 h3 b4 a1 a2 h1 h4 d3 e5 e4 f3 d4 b2 f2 c3 c4 b1 g3 d5 f1 g4 c5 g2 b3 e1 c1 b5 c2 h2 h5 d2 f4 a5 g5 g1 f5 e2 a3 e3 a4
```

Block 4: knight then bishop (awkward shapes), ranks 4–5 (16). Swap at 8.

```
.venv/bin/python chess/record_pick.py --dataset so101_pick3 --resume --episodes 16 --swap-every 8 g4 h5 f5 e5 c4 d4 f4 b5 g5 e4 a4 b4 c5 d5 h4 a5
```

Coverage after all four blocks: every taught square at least twice, ranks 4–5
four times, pawns 61 of 116 episodes.

Stop any block with `q` and continue later with the same command plus `--resume`
and the remaining squares (the first block needs `--resume` too once it has saved
anything).

## Then

```
DATASET=so101_pick3 sh train_chess_gpu.sh start      # from scratch, default steps
```

Test with `chess/run_pick.py ... --out act_so101_pick3` on d2, a1, g4 and a rank-5 square.
