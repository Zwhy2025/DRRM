## Kuavo `kuavo_task1_1-200` 训练指引

### 数据信息
- 路径：`datasets/kuavo_task1_1-200/`
- 任务：`Pick and Place`（见 `lerobot/meta/tasks.jsonl`）
- 观测/动作：
  - `observation.state` / `action`: 16 维（双臂 + 夹爪）
  - 图像：`observation.images.head_cam_h`、`observation.images.wrist_cam_l`，形状 `[3,480,640]`

### 推荐配置
- 训练配置：`configs/vodp_train/vodp_kuavo_task1.yaml`
- 加速配置：`configs/accelerate_config.yaml`（按 GPU 数调整 `gpu_ids` / `num_processes`）
- HF repo_id 已在配置中固定为 `local-dataset`，无需额外覆盖。

### 运行命令
```bash
conda activate drrm

dataset=datasets/kuavo_task1_1-200
task="Pick and Place"
demo=null                 # 或设为整数限制样本数
config_dir=configs/vodp_train/vodp_kuavo_task1.yaml

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
- `dataset/task` 与 `tasks.jsonl` 中一致。
- `features` 与 Parquet 列名/形状匹配；有 `head_cam_h` 和 `wrist_cam_l` 两个相机键。
- 显存不足可调低 `train_batch_size` 或将 `accelerate_config` 的 `num_processes` 设 1。***

