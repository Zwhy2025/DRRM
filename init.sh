#!/usr/bin/env bash

# 初始化/更新 DRRM 环境的脚本
# 目标：可重复执行、失败即停、日志清晰

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR"
ENV_NAME="${ENV_NAME:-drrm}"
THIRD_PARTY_DIR="$WS_DIR/third_party"
VGGT_REPO="https://github.com/facebookresearch/vggt.git"
VGGT_DIR="$THIRD_PARTY_DIR/vggt"
ROBOTWIN_DIR="$WS_DIR/simulation/robotwin"
NUMPY_PIN="${NUMPY_PIN:-numpy<1.27}"
PYTORCH3D_TAG="${PYTORCH3D_TAG:-v0.7.7}"
PYTORCH3D_GIT="${PYTORCH3D_GIT:-git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_TAG}}"
PYTORCH3D_GIT_MIRROR="${PYTORCH3D_GIT_MIRROR:-git+https://ghproxy.com/https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_TAG}}"
PYTORCH3D_ZIP="${PYTORCH3D_ZIP:-https://github.com/facebookresearch/pytorch3d/archive/refs/tags/${PYTORCH3D_TAG}.zip}"
APT_UPDATED=0
SUDO="${SUDO:-sudo}"

# 若脚本已以 root 运行，避免重复使用 sudo
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run() {
  log "→ $*"
  "$@"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "缺少命令: $1"
    exit 1
  fi
}

maybe_apt_update() {
  if [ "$APT_UPDATED" -eq 0 ]; then
    run ${SUDO:+$SUDO }apt update
    APT_UPDATED=1
  fi
}

ensure_apt_packages() {
  local pkgs=("$@")
  local missing=()
  for pkg in "${pkgs[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    maybe_apt_update
    run ${SUDO:+$SUDO }apt install -y "${missing[@]}"
  else
    log "APT 包已满足: ${pkgs[*]}"
  fi
}

ensure_conda_env() {
  require_cmd conda
  # 允许非交互式激活
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    log "复用已有 conda 环境: $ENV_NAME"
  else
    run conda create -y -n "$ENV_NAME" python=3.10
  fi
  run conda activate "$ENV_NAME"
}

install_base_project() {
  log "安装/更新项目可编辑依赖"
  run pip install "$NUMPY_PIN"
  run pip install -e "$WS_DIR"
}

install_vggt() {
  mkdir -p "$THIRD_PARTY_DIR"
  if [ -d "$VGGT_DIR/.git" ]; then
    log "vggt 已存在，跳过 clone"
  else
    run git clone "$VGGT_REPO" "$VGGT_DIR"
  fi
  pushd "$VGGT_DIR" >/dev/null
  run pip install "$NUMPY_PIN"
  run pip install .
  popd >/dev/null
}

install_pytorch3d() {
  # 多路回退，避免 git TLS / 分支不可用
  try_install() {
    local label="$1"; shift
    log "尝试安装 pytorch3d: $label"
    if run "$@"; then
      return 0
    fi
    log "pytorch3d 安装失败: $label"
    return 1
  }

  if try_install "官方 Git $PYTORCH3D_TAG" pip install --no-build-isolation --no-deps "$PYTORCH3D_GIT"; then
    return 0
  fi
  if try_install "GH 代理" pip install --no-build-isolation --no-deps "$PYTORCH3D_GIT_MIRROR"; then
    return 0
  fi
  if try_install "ZIP 包" pip install --no-build-isolation --no-deps "$PYTORCH3D_ZIP"; then
    return 0
  fi

  log "pytorch3d 安装多次失败，请检查网络或手动安装。"
  return 1
}

install_robotwin_deps() {
  ensure_apt_packages \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    ffmpeg \
    parallel

  pushd "$ROBOTWIN_DIR" >/dev/null
  run pip install \
    sapien==3.0.0b1 \
    scipy==1.10.1 \
    mplib==0.1.1 \
    trimesh==4.4.3 \
    open3d==0.18.0 \
    openai
  run pip install "$NUMPY_PIN"
  install_pytorch3d
  popd >/dev/null
}

main() {
  log "工作空间: $WS_DIR"
  ensure_conda_env
  install_base_project
  install_vggt
  install_robotwin_deps
  run pip install h5py loguru
  log "完成 ✅"
}

main "$@"