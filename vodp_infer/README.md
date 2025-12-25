# VODP 实机最小运行器

该目录提供一个轻量的实机推理流程：
- `vodp_server.py`: TCP 模型服务端（VODP 推理）。
- `vodp_client.py`: 客户端，使用本目录 `datacenter.py` 获取观测并下发动作。

## 快速开始

1) 编辑 `config.yaml`，设置：
- `model.checkpoint_dir`
- `mapping.cameras`（datacenter 相机名 -> 模型输入键）
- `mapping.model_arm_order` 与 DOF

2) 如需调整相机/机械臂话题，请修改 `config_datacenter.yaml`。

3) 启动服务端：
```bash
python3 vodp_server.py --config config.yaml
```

4) 启动客户端：
```bash
python3 vodp_client.py --config config.yaml
```

## 说明
- `mapping.expected_size` 需与相机输出一致；如需缩放，设置 `resize: true`（依赖 `cv2`）。
- 单臂部署时，在 `config_datacenter.yaml` 中关闭另一只手臂，并同步调整 `mapping.model_arm_order`。
- 如果策略输出动作序列，`execute_horizon: false` 将只执行第一步。
