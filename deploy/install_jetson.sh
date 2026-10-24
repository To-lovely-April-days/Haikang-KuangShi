#!/usr/bin/env bash
# Jetson 一键安装：复制代码到 /opt/oresorter、建 venv、装依赖（优先本地 wheels/ 离线）、建数据目录、装 systemd 单元。
# 用法（在仓库 backend/ 目录下）：  sudo ./deploy/install_jetson.sh [--user <服务运行用户>] [--no-enable]
# 幂等：可重复执行升级代码；不会覆盖已存在的 /opt/oresorter/config/jetson.yaml 与 /data/oresorter。
set -euo pipefail

INSTALL_DIR=/opt/oresorter
DATA_DIR=/data/oresorter
SERVICE_USER="${SUDO_USER:-$(id -un)}"
ENABLE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --user) SERVICE_USER="$2"; shift 2 ;;
    --no-enable) ENABLE=0; shift ;;
    *) echo "未知参数 $1"; exit 2 ;;
  esac
done
SERVICE_GROUP=$(id -gn "$SERVICE_USER")
SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)

log() { echo -e "\033[1;32m[install]\033[0m $*"; }
warn() { echo -e "\033[1;33m[install]\033[0m $*"; }

[ "$(id -u)" -eq 0 ] || { echo "请用 sudo 运行"; exit 1; }
[ "$(uname -m)" = "aarch64" ] || warn "当前架构 $(uname -m) 不是 aarch64，确认是在 Jetson 上？"
if [ -f /etc/nv_tegra_release ]; then log "L4T: $(head -1 /etc/nv_tegra_release)"; else warn "未检测到 /etc/nv_tegra_release（非 JetPack？）"; fi

PY=$(command -v python3.10 || command -v python3)
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || { echo "需要 Python 3.10（JetPack 6 自带）"; exit 1; }

log "安装系统依赖"
apt-get update -qq || warn "apt update 失败（离线？继续）"
apt-get install -y -qq python3-venv python3-pip python3-dev rsync ethtool >/dev/null 2>&1 || warn "apt 安装部分失败（离线时请确认 python3-venv 已装）"

log "同步代码到 $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
rsync -a --delete \
  --exclude .venv --exclude data --exclude .git --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache \
  --exclude 'config/jetson.yaml' \
  "$SRC_DIR"/ "$INSTALL_DIR"/
# 配置文件只在不存在时复制，保留现场修改
[ -f "$INSTALL_DIR/config/jetson.yaml" ] || cp "$SRC_DIR/config/jetson.yaml" "$INSTALL_DIR/config/jetson.yaml"

log "创建虚拟环境（--system-site-packages 以复用 JetPack 自带的 tensorrt / cuda python 绑定）"
if [ ! -x "$INSTALL_DIR/.venv/bin/python" ]; then
  "$PY" -m venv --system-site-packages "$INSTALL_DIR/.venv"
fi
VPIP="$INSTALL_DIR/.venv/bin/pip"
"$VPIP" install -q --upgrade pip wheel setuptools || warn "pip 升级失败（离线？继续）"

log "安装 Python 依赖"
if [ -d "$SRC_DIR/wheels" ] && ls "$SRC_DIR/wheels"/*.whl >/dev/null 2>&1; then
  log "发现本地 wheels/，离线安装"
  "$VPIP" install -q --no-index --find-links "$SRC_DIR/wheels" -r "$INSTALL_DIR/requirements-jetson.txt" \
    || { warn "离线安装失败，尝试在线补齐"; "$VPIP" install -q --find-links "$SRC_DIR/wheels" -r "$INSTALL_DIR/requirements-jetson.txt"; }
else
  warn "没有 wheels/ 目录：PyTorch 将从 PyPI 安装，这在 Jetson 上通常不可用。请按 docs/01 §10 下载 NVIDIA 专用 wheel 放到 backend/wheels/ 后重跑。"
  "$VPIP" install -q -r "$INSTALL_DIR/requirements-jetson.txt" || warn "依赖安装未完全成功，请检查"
fi
"$VPIP" install -q -e "$INSTALL_DIR" --no-deps || true

log "数据目录 $DATA_DIR"
mkdir -p "$DATA_DIR"/{models,snapshots,previews,background,logs,captures}
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR" "$INSTALL_DIR"

log "系统调优文件"
install -m 644 "$INSTALL_DIR/deploy/sysctl-oresorter.conf" /etc/sysctl.d/90-oresorter.conf
sysctl -q -p /etc/sysctl.d/90-oresorter.conf || warn "sysctl 应用失败"
chmod +x "$INSTALL_DIR/deploy/tune_jetson.sh"

log "systemd 单元"
sed -e "s/__USER__/$SERVICE_USER/" -e "s/__GROUP__/$SERVICE_GROUP/" "$INSTALL_DIR/deploy/oresorter.service" > /etc/systemd/system/oresorter.service
install -m 644 "$INSTALL_DIR/deploy/oresorter-tune.service" /etc/systemd/system/oresorter-tune.service
systemctl daemon-reload
if [ "$ENABLE" -eq 1 ]; then
  systemctl enable oresorter-tune.service oresorter.service >/dev/null
  systemctl restart oresorter-tune.service || true
  systemctl restart oresorter.service
  sleep 2
  systemctl --no-pager --lines=5 status oresorter.service || true
fi

cat <<MSG

安装完成。后续检查清单：
  1. 海康 MVS SDK：dpkg -l | grep -i mvs ；/opt/MVS/lib/aarch64 存在；
  2. 网络：参照 deploy/netplan-example.yaml 配两网口静态 IP 并 netplan apply；ping 192.168.10.10（相机）、192.168.1.30（下位机）；
  3. 配置：编辑 $INSTALL_DIR/config/jetson.yaml（相机 IP/序列号、下位机地址）后 systemctl restart oresorter；
  4. 验证：curl http://127.0.0.1:8000/api/v1/health ；journalctl -u oresorter -f ；
  5. 模型引擎：训练后首次部署会自动调用 trtexec 生成 model.plan（约 30~90 s）。
MSG
