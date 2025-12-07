# D-robotics 机器人操作平台
## 🔥 最新工作: VO-DP
[![project page](https://img.shields.io/badge/Project%20Page-DRRM-orange)](https://d-robotics-ai-lab.github.io/vodp/)
[![arXiv paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.15530)
[![dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/D-Robotics/DRRM)

https://github.com/user-attachments/assets/fdca37aa-164b-4281-a446-3c909a3f1456

## 🌟 更新日志
- **2025-11**: robotwin2.0 模拟器现已可用！查看 [simulation/robotwin2.0](simulation/robotwin2.0/)
- **2025-10**: 我们发布了最新工作 [VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation](https://arxiv.org/abs/2510.15530) 的代码和数据集。

## ⚙️ 安装
### 基础环境配置
```bash
git clone https://github.com/D-Robotics-AI-Lab/DRRM.git
cd DRRM
conda create -n drrm python=3.10
conda activate drrm
pip install -e .
```

### VODP 环境配置

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install .
cd ../..
```

### Robotwin 环境配置
详细模拟器安装和使用步骤请参考 `simulation/robotwin/README.md`。

## 📊 数据集准备
创建 `datasets/` 目录并从 HuggingFace 下载预处理的数据集包：

```bash
mkdir -p datasets
```

可用的预处理包（选择一个）：
- [`drrm_robotwin1.0_D435_200_rgb`](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_rgb) — 仅 RGB 版本（无点云）
- [`drrm_robotwin1.0_D435_200_pcd`](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_pcd) — 包含点云

访问 https://huggingface.co/datasets/D-Robotics/DRRM 下载数据集并放置在 `datasets/` 目录下。

如需使用自有数据训练，请参阅 `[自定义数据训练指引](docs/custom_training.zh.md)`。
预置数据示例训练指南：
- `kuavo_task1_1-200`: `docs/training_kuavo_task1_1-200.md`
- `ur12e/real_libero_spatial`: `docs/training_ur12e.md`

## 📑 训练
1. 根据你的训练环境修改加速配置文件：[configs/accelerate_config.yaml](configs/accelerate_config.yaml)
2. 在训练脚本 [scripts/train_demo.sh](scripts/train_demo.sh) 中指定以下参数：
   - `dataset`: 训练数据集路径
   - `task`: 训练任务
   - `demo`: 使用的演示样本数量（设置为 `null` 表示无限制）
   - `config_dir`: 训练配置目录（参考 [configs/](configs/)）

- 训练 VODP
```
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/vodp_train" \
    --config-name="vodp_23d_1f.yaml" \
    train_dataset.path=datasets/lerobot_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

- 训练 DP
```
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/dp_train" \
    --config-name="dp.yaml" \
    train_dataset.path=datasets/lerobot_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

- 训练 DP3
```
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/dp3_train" \
    --config-name="dp3.yaml" \
    train_dataset.path=datasets/lerobot43d_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

<!-- ### Training Your Own Model
...... -->

## 🤖 模拟评估
DRRM 兼容 Robotwin 模拟器 — 请参考以下文件：

- `scripts/eval/eval_robotwin.sh` — 用于实验的便捷 shell 包装脚本。
- `scripts/eval/robotwin_exp/` — 论文中使用的示例实验包装脚本。
- `simulation/robotwin/script/` — 面向模拟器的 Python 评估脚本（例如 `eval_policy_vodp.py`、`eval_policy_dp.py`、`eval_policy_dp3.py`）。

支持的基准任务：

```
[
    'block_hammer_beat', 'bottle_adjust', 'container_place',
    'dual_bottles_pick_hard', 'put_apple_cabinet',
    'tool_adjust', 'pick_apple_messy', 'dual_bottles_pick_easy',
    'diverse_bottles_pick', 'empty_cup_place', 'shoe_place',
    'dual_shoes_place', 'blocks_stack_easy', 'block_handover'
]
```


### 基本用法
将 `YOUR/CHECKPOINT/DIR` 和 `YOUR/SAVE/DIR` 替换为你的检查点目录路径和要存储评估结果的目录路径。`--num-process` 标志控制并行模拟器工作进程数。

```bash
# 评估 VODP
python simulation/robotwin/script/eval_policy_vodp.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

```bash
# 评估 DP
python simulation/robotwin/script/eval_policy_dp.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

```bash
# 评估 DP3
python simulation/robotwin/script/eval_policy_dp3.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

## 👏 引用
```
@article{ni2025vodp,
  title={VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation},
  author={Zehao Ni and Yonghao He and Lingfeng Qian and Jilei Mao and Fa Fu and Wei Sui and Hu Su and Junran Peng and Zhipeng Wang and Bin He},
  journal={arXiv preprint arXiv:2510.15530},
  year={2025}
}
```

