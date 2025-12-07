## 自定义数据训练指引

面向想用 DRRM 训练自己数据的快速步骤，假设你已按 `init.sh` 配好环境。

### 1. 准备数据目录

推荐放在 `datasets/<your_dataset>/`，沿用 lerobot 约定：

```
datasets/<your_dataset>/
  lerobot/
    meta/
      info.json       # 数据集全局信息
      tasks.jsonl     # 每行一个任务名称/索引
    data/
      chunk-000/
        episode_000000.parquet
        ...
    videos/ (可选，如有视频)
```

- `info.json` 关键字段示例（来自现有 `kuavo_task1_1-200`）：
  - `robot_type`: 机器人型号（示例 `kuavo4pro`）
  - `total_episodes` / `splits.train`: 训练集范围（如 `"0:200"`）
  - `data_path`: 样本相对路径模板，例如 `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
  - `features`: 描述观测/动作键、dtype、shape（需与 Parquet 实际列一致）
- `tasks.jsonl`: 每行一个 JSON，如 `{"task_index":0,"task":"Pick and Place"}`；`train_dataset.task` 将使用此名字。
- Parquet 内容需包含：
  - `observation.*`（如 `observation.images.head_cam_h`、`observation.state` 等）与 `action` 对齐
  - `timestamp/frame_index/episode_index/task_index` 等索引字段保持一致

快速自检：
- 任务名是否出现在 `tasks.jsonl`
- Parquet 是否能被 pandas/pyarrow 读取且列名匹配 `features`
- 观测与动作长度一致，帧数大于 demo 数

### 2. 配置训练命令

可直接使用 `scripts/train_demo.sh`，或在命令行写参数：

```bash
dataset=datasets/<your_dataset>
task="Pick and Place"          # 对应 tasks.jsonl 的 task 字段
demo=null                      # null=用全部数据；或填整数截取 demo 数
config_dir=configs/vodp_train/vodp_23d_1f.yaml  # 或其它配置

accelerate launch \
  --config_file configs/accelerate_config.yaml \
  main.py \
  --config-path="${config_dir%/*}" \
  --config-name="${config_dir##*/}" \
  train_dataset.path=$dataset \
  train_dataset.task="$task" \
  train_dataset.demo=$demo
```

常用入口：
- VODP: `configs/vodp_train/vodp_23d_1f.yaml`
- DP/DP3: 参考 `configs/dp_train/`、`configs/dp3_train/`

`configs/accelerate_config.yaml` 控制分布式/GPU 数；如单卡可保留 `gpu_ids: "0"`，`num_processes: 1`。

### 3. 训练前 Checklist

- 环境：`conda activate drrm`，`pip show vggt`，`python -c "import torch"` 正常
- 数据：`ls datasets/<your_dataset>/lerobot/meta/{info.json,tasks.jsonl}`
- 配置：`task` 与 `tasks.jsonl` 匹配；`demo` 不大于可用样本
- 存储：`output_dir`（配置文件内）所在路径有写权限

### 4. 常见问题与排查

- **任务名不匹配**：报找不到任务时，核对 `train_dataset.task` 与 `tasks.jsonl` 中 `task` 字段完全一致。
- **维度/键错误**：报缺少 `observation.xxx` 或 shape 不符，检查 `info.json` 的 `features` 与 Parquet 列名、形状是否一致。
- **数据不足**：`demo` 过大导致越界，先将 `demo` 设为 `null` 或更小数字。
- **多模态相机键**：配置文件的 `shape_meta.obs` 中相机键（如 `head_cam`、`front_cam`）需与数据内键对应；如使用自定义键，需在配置里同步修改。
- **显存/进程数**：OOM 时调小 batch 或在 `configs/accelerate_config.yaml` 将 `num_processes` 设为 1。

### 5. 参考现有数据示例

- `datasets/kuavo_task1_1-200/lerobot/meta/info.json` 与 `tasks.jsonl` 提供了完整示例，可仿照填充字段与目录结构。

