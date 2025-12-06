# RobotWin — 安装与配置

本文档描述了设置本项目使用的 RobotWin 仿真环境的最小步骤。

## 概述

- 安装系统 Vulkan 驱动和工具
- 创建并激活 conda 环境
- 安装 Python 依赖（特定测试版本）
- 下载所需资源
- 对 mplib 规划器进行两处小的本地修改以避免运行时错误

## 前置要求

安装 Vulkan 及相关驱动：
```bash
sudo apt update
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
sudo apt install ffmpeg
```

## Python 环境

激活你的 conda 环境（如果环境名不同，请将 `drrm` 替换为你的环境名）：
```bash
conda activate drrm
```

安装所需的 Python 包（本项目使用的版本）：
```bash
pip install sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 trimesh==4.4.3 open3d==0.18.0 openai
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## 下载资源

从 robotwin 根目录创建 assets 目录并下载：
```bash
cd simulation/robotwin
mkdir -p assets
cd assets
python ../script/download_asset.py
```

解压下载的 zip 文件：
```bash
unzip aloha_urdf.zip && rm -f aloha_urdf.zip
unzip main_models.zip && rm -f main_models.zip
```

## mplib 所需的本地修改

对 mplib 进行两处小的修改以避免兼容性问题。编辑你的 site-packages 或本地副本中的 `mplib/planner.py`。

1) 创建 ArticulatedModel 时移除 `convex=True` 参数（约第 71 行）：
修改前：
```py
self.robot = ArticulatedModel(
    urdf,
    srdf,
    [0, 0, -9.81],
    user_link_names,
    user_joint_names,
    convex=True,
    verbose=False,
)
```
修改后：
```py
self.robot = ArticulatedModel(
    urdf,
    srdf,
    [0, 0, -9.81],
    user_link_names,
    user_joint_names,
    # convex=True,
    verbose=False,
)
```

2) 从螺旋规划失败条件中移除 `or collide`（约第 848 行）：
修改前：
```py
if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
    return {"status": "screw plan failed"}
```
修改后：
```py
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
    return {"status": "screw plan failed"}
```


