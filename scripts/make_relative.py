"""Copy a pick dataset with the arm joints made relative to each episode's first frame.

  .venv/bin/python scripts/make_relative.py so101_pick3_wrist_v4 so101_pick3_wrist_v4_rel

In the copy, the five arm joints of observation.state and action have the episode's first
observation.state subtracted; the gripper stays absolute. A policy trained on it cannot tell
which square it is over from its joints, so the wrist camera has to find the piece.
run_pick.py feeds such a policy (dataset name ending in _rel) the same way: joints minus the
reading when the policy starts, and adds that reading back to its commands.
Statistics (meta/stats.json and the per-episode ones) are recomputed for the two features.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FEATS = ("observation.state", "action")
QS = {"q01": 1, "q10": 10, "q50": 50, "q90": 90, "q99": 99}


def stats(x):
    s = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.array([len(x)])}
    s.update({k: np.percentile(x, q, axis=0) for k, q in QS.items()})
    return s


def main():
    root = Path(__file__).parent.parent / "datasets"
    src, dst = root / sys.argv[1], root / sys.argv[2]
    shutil.copytree(src, dst)
    data = {f: [] for f in FEATS}
    per_ep = {}
    start = {}  # episode -> first arm joints
    files = sorted((dst / "data").rglob("*.parquet"))
    for path in files:  # episodes are whole within a file and in frame order
        t = pq.read_table(path)
        ep = t["episode_index"].to_numpy()
        arr = {f: np.stack(t[f].to_numpy(zero_copy_only=False)).astype(np.float32) for f in FEATS}
        for e in np.unique(ep):
            m = ep == e
            assert t["frame_index"].to_numpy()[m][0] == 0, f"episode {e} does not start in {path.name}"
            start[e] = arr["observation.state"][m][0, :5].copy()
            for f in FEATS:
                arr[f][m, :5] -= start[e]
                per_ep.setdefault(e, {})[f] = stats(arr[f][m].astype(np.float64))
        for f in FEATS:
            i = t.schema.get_field_index(f)
            t = t.set_column(i, t.schema.field(i), pa.FixedSizeListArray.from_arrays(pa.array(arr[f].ravel(), pa.float32()), 6))
            data[f].append(arr[f])
        pq.write_table(t, path)

    stats_path = dst / "meta" / "stats.json"
    all_stats = json.load(open(stats_path))
    for f in FEATS:
        all_stats[f] = {k: v.tolist() for k, v in stats(np.concatenate(data[f]).astype(np.float64)).items()}
    json.dump(all_stats, open(stats_path, "w"), indent=4)

    for path in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(path)
        eps = t["episode_index"].to_pylist()
        for f in FEATS:
            for k in list(QS) + ["min", "max", "mean", "std", "count"]:
                i = t.schema.get_field_index(f"stats/{f}/{k}")
                t = t.set_column(i, t.schema.field(i), pa.array([per_ep[e][f][k].tolist() for e in eps], t.schema.field(i).type))
        pq.write_table(t, path)

    info_path = dst / "meta" / "info.json"
    info = json.load(open(info_path))
    info["relative_to_first_frame"] = True
    json.dump(info, open(info_path, "w"), indent=4)
    print(f"{dst.name}: {len(start)} episodes, {sum(len(a) for a in data['action'])} frames")
    for f in FEATS:
        print(f"  {f} mean {np.round(all_stats[f]['mean'], 1).tolist()} std {np.round(all_stats[f]['std'], 1).tolist()}")


if __name__ == "__main__":
    main()
