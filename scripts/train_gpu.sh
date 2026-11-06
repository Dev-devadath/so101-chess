#!/bin/sh
# Train the chess policy on the GPU box instead of the Mac. POLICY=act (default) or smolvla.
#   sh scripts/train_gpu.sh start    copy datasets/<DATASET> to the box (if not there yet) and launch training (nohup)
#   POLICY=smolvla sh scripts/train_gpu.sh start   fine-tune lerobot/smolvla_base instead (needs the HF cache pushed, see below)
#   sh scripts/train_gpu.sh log      tail the training log on the box
#   sh scripts/train_gpu.sh pull     copy the checkpoints back to outputs/<POLICY>_<DATASET>
#   sh scripts/train_gpu.sh resume   resume from the last checkpoint on the box (after a crash/reboot)
#   sh scripts/train_gpu.sh autoresume   install a @reboot cron job on the box that runs the resume
# GPU_HOST is an ssh target (default: the "gpu" alias in ~/.ssh/config; the box's
# LAN IP changes, so override with e.g. GPU_HOST=tinkerspace@192.168.11.67).
# Box layout: ~/lerobot-arm with ./venv (lerobot 0.5.x, cuda); datasets/ and outputs/ under it.
# Transfers use tar over ssh: macOS openrsync crashes (exit 11) against the box rsync 3.2 on video files.
# The box's internet is ~15 kB/s, so HF_HUB_OFFLINE=1: smolvla_base and the SmolVLM2 processor files
# must already be in its ~/.cache/huggingface/hub (pushed from the Mac's cache with tar over ssh).
. "$(dirname "$0")/chess_act_env.sh"
GPU_HOST=${GPU_HOST:-gpu}
REMOTE=lerobot-arm   # relative to the login home dir; a local ~ would expand to the Mac path
POLICY=${POLICY:-act}
case "$POLICY" in
  act)     RUN=act_$DATASET; LOG=train_$DATASET.log
           POLICY_ARGS="--policy.type=act --policy.use_amp=true"; DEF_STEPS=80000 ;;
  smolvla) RUN=smolvla_$DATASET; LOG=train_$RUN.log; OUT_DIR=outputs/$RUN
           POLICY_ARGS="--policy.path=lerobot/smolvla_base --policy.load_vlm_weights=false --rename_map='$SMOLVLA_RENAME_MAP'"; DEF_STEPS=20000 ;;
           # load_vlm_weights: the checkpoint already holds the VLM weights; true would fetch 2 GB from the hub
  *) echo "POLICY must be act or smolvla" >&2; exit 2 ;;
esac

case "${1:-start}" in
  start)
    if ssh $GPU_HOST "test -d $REMOTE/datasets/$DATASET"; then echo "dataset already on the box, not copying"; else
      COPYFILE_DISABLE=1 tar --no-xattrs -C datasets -cf - $DATASET | ssh $GPU_HOST "mkdir -p $REMOTE/datasets && tar -C $REMOTE/datasets -xf -" || exit 1
    fi
    ssh $GPU_HOST "cd $REMOTE && HF_HUB_OFFLINE=1 nohup ./venv/bin/lerobot-train \
      $POLICY_ARGS --policy.device=cuda --policy.push_to_hub=false \
      --dataset.repo_id=local/$DATASET --dataset.root=datasets/$DATASET --dataset.video_backend=pyav \
      --output_dir=$OUT_DIR --job_name=$RUN \
      --steps=${STEPS:-$DEF_STEPS} --batch_size=${BATCH:-8} --save_freq=${SAVE_FREQ:-10000} \
      --log_freq=200 --eval_freq=0 > $LOG 2>&1 & echo started pid \$!"
    ;;
  log)  ssh $GPU_HOST "tail -n ${LINES:-20} $REMOTE/$LOG" ;;
  resume) ssh $GPU_HOST "sh $REMOTE/resume_$RUN.sh" ;;
  autoresume)
    # Remote script: resume only if a checkpoint exists and no trainer for this run is alive.
    # Resuming a finished run is a no-op (lerobot exits at once), so it is safe to leave installed.
    ssh $GPU_HOST "cat > $REMOTE/resume_$RUN.sh" <<REMOTE_EOF
#!/bin/sh
# Installed by train_chess_gpu.sh autoresume. Run by cron at boot, or by hand.
cd ~/$REMOTE || exit 1
CFG=$OUT_DIR/checkpoints/last/pretrained_model/train_config.json
[ -f "\$CFG" ] || { echo "no checkpoint yet, nothing to resume"; exit 0; }
pgrep -f "$OUT_DIR" >/dev/null && { echo "trainer already running"; exit 0; }
HF_HUB_OFFLINE=1 nohup ./venv/bin/lerobot-train --config_path=\$CFG --resume=true >> $LOG 2>&1 &
echo "resumed, pid \$!"
REMOTE_EOF
    ssh $GPU_HOST "systemctl is-active cron >/dev/null || { echo 'cron is not running on the box' >&2; exit 1; }
      (crontab -l 2>/dev/null | grep -v resume_$RUN.sh; echo '@reboot sleep 60 && sh \$HOME/$REMOTE/resume_$RUN.sh >> \$HOME/$REMOTE/resume_$RUN.log 2>&1') | crontab -
      echo installed:; crontab -l | grep resume_$RUN.sh"
    ;;
  pull) mkdir -p $OUT_DIR && ssh $GPU_HOST "tar -C $REMOTE/$OUT_DIR -cf - checkpoints" | tar -C $OUT_DIR -xf - && ls $OUT_DIR/checkpoints ;;
  *) echo "usage: $0 start|log|pull|resume|autoresume" >&2; exit 2 ;;
esac
