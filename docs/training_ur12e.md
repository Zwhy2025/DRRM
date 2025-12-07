## UR12e `real_libero_spatial` 训练指引

### 数据信息
- 路径：`datasets/ur12e/real_libero_spatial/`
- 任务：`pick up the green bowl and place it on the plate`（见 `meta/tasks.jsonl`）
- 观测/动作：
  - `observation.state` / `action`: 7 维（6 关节 + gripper）
  - 图像：`observation.images.image`、`observation.images.wrist_image`，形状 `[3,480,640]`
  - 帧率 30 fps，98 集

### 推荐配置
- 训练配置：`configs/vodp_train/vodp_ur12e.yaml`
- 加速配置：`configs/accelerate_config.yaml`
- HF repo_id 已在配置中固定为 `local-dataset`，无需额外覆盖。

### 运行命令
```bash
conda activate drrm

dataset=datasets/ur12e/real_libero_spatial
task="pick up the green bowl and place it on the plate"
demo=null                 # 或设为整数限制样本数
config_dir=configs/vodp_train/vodp_ur12e.yaml

accelerate launch \
  --config_file configs/accelerate_config.yaml \
  main.py \
  --config-path="${config_dir%/*}" \
  --config-name="${config_dir##*/}" \
  train_dataset.path=$dataset \
  train_dataset.task="$task" \
  train_dataset.demo=$demo
```

### 检查项
- `task` 字符串与 `tasks.jsonl` 完全一致。
- 数据列与 `info.json` 一致，包含 `image` / `wrist_image` 两路相机、7 维 state/action。
- 如显存不足，降低 `train_batch_size` 或在 `configs/accelerate_config.yaml` 将 `num_processes` 设 1。

