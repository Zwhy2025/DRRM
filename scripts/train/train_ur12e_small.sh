#!/usr/bin/env bash
set -euo pipefail

# 确保可用 conda 并激活环境
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  # 兜底初始化
  eval "$(conda shell.bash hook)"
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "drrm" ]; then
  conda activate drrm
fi

dataset=datasets/ur12e_small
task='real_libero_spatial'
demo=50                 # 或设为整数限制样本数
config_dir=configs/vodp_train/vodp_ur12e.yaml

accelerate launch \
  --config_file configs/accelerate_config.yaml \
  main.py \
  --config-path="${config_dir%/*}" \
  --config-name="${config_dir##*/}" \
  train_dataset.path=$dataset \
  train_dataset.task=$task \
  train_dataset.demo=$demo
