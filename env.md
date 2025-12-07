# 数据转换
sudo apt install parallel
pip install h5py loguru

# 拉取数据
```
# (不行版本不兼容) 
pip install -U huggingface_hub
# 使用特定版本 这个默认已经安装过了
pip install huggingface-hub-0.36.0
```

## robotwin仿真环境配置
使用这个安装不会报错
```
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```