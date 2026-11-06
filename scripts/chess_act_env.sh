# Shared settings for the chess pick-and-place ACT pipeline.
# Sourced by scripts/train_gpu.sh. Override anything with env vars, e.g.  DATASET=so101_pick4 sh scripts/train_gpu.sh start
cd "$(dirname "$0")/.." || exit 1   # repo root

FOLLOWER_PORT=/dev/tty.usbmodem5B3D0453051
LEADER_PORT=/dev/tty.usbmodem5B610337631

DATASET=${DATASET:-so101_pick3_clean}
TASK=${TASK:-"pick up the chess piece and place it in the box"}
OUT_DIR=outputs/act_${DATASET}
# smolvla_base expects observation.images.camera1/2/3; map our two cams onto the first two (no spaces: passed unquoted)
SMOLVLA_RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# macOS reshuffles camera indices whenever a camera (re)connects, so resolve
# them by fingerprint at launch (see chess/cameras.py). Override with
# WRIST_INDEX=<n> / TOP_INDEX=<n> if the fingerprint lookup is wrong.
cam_index() {
  .venv/bin/python -c "import sys; sys.path.insert(0, 'chess'); from cameras import camera_index; print(camera_index('$1'))" 2>/dev/null
}
# Only record/eval need cameras; the train scripts source this file too, so look them up on demand.
need_cameras() {
  WRIST_INDEX=${WRIST_INDEX:-$(cam_index wrist)}
  TOP_INDEX=${TOP_INDEX:-$(cam_index overhead)}
  if [ -z "$WRIST_INDEX" ] || [ -z "$TOP_INDEX" ]; then
    echo "could not find both cameras (wrist=$WRIST_INDEX top=$TOP_INDEX); run: .venv/bin/lerobot-find-cameras opencv" >&2
    exit 1
  fi
  CAMERAS="{ wrist: {type: opencv, index_or_path: $WRIST_INDEX, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: $TOP_INDEX, width: 640, height: 480, fps: 30} }"
  echo "cameras: wrist=$WRIST_INDEX top=$TOP_INDEX | dataset=$DATASET"
}
