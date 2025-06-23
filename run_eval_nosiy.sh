cut_ratios=(0.1 0.15 0.2 0.25 0.3 0.4 0.5 )
seed=2025

for ratio in "${cut_ratios[@]}"; do
  echo "==== Running with cut_ratio=${ratio} ===="
  /workspace/.venv/bin/python -m evaluation.eval_nosiy \
    --dataset_dir "data" \
    --resume 1 \
    --resume_model "pretrain/gpv_pose_update.pth" \
    --model_save "eval_logs" \
    --draw_gt=false \
    --our_camK=false \
    --cut=true \
    --cut_ratio "${ratio}" \
    --cut_repeat=1 \
    --cut_method=two_plane \
    --cut_seed=${seed}
done

