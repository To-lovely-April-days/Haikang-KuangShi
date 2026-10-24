#!/usr/bin/env bash
# Jetson 开机调优：最高性能模式、锁频、相机网口巨帧、接收缓冲。
# 由 oresorter-tune.service 以 root 运行；也可手动执行。所有步骤失败不致命（记录后继续）。
set -u
CAM_IF="${ORESORTER_CAM_IF:-}"          # 相机直连网口名；为空时按 192.168.10.x 地址自动查找
log() { echo "[tune] $*"; }

if command -v nvpmodel >/dev/null 2>&1; then
  nvpmodel -m 0 && log "nvpmodel -m 0 OK" || log "nvpmodel 失败（忽略）"
fi
if command -v jetson_clocks >/dev/null 2>&1; then
  jetson_clocks && log "jetson_clocks OK" || log "jetson_clocks 失败（忽略）"
fi

if [ -z "$CAM_IF" ]; then
  CAM_IF=$(ip -o -4 addr show | awk '/192\.168\.10\./ {print $2; exit}')
fi
if [ -n "$CAM_IF" ]; then
  ip link set dev "$CAM_IF" mtu 9000 && log "$CAM_IF mtu 9000 OK" || log "$CAM_IF 设置 MTU 失败（交换机/网卡可能不支持巨帧）"
  # 增大网卡环形缓冲（若驱动支持）
  if command -v ethtool >/dev/null 2>&1; then
    ethtool -G "$CAM_IF" rx 4096 2>/dev/null && log "$CAM_IF rx ring 4096" || true
  fi
else
  log "未找到相机网口（192.168.10.x），跳过 MTU 设置"
fi

if [ -f /etc/sysctl.d/90-oresorter.conf ]; then
  sysctl -q -p /etc/sysctl.d/90-oresorter.conf && log "sysctl OK" || log "sysctl 失败（忽略）"
fi
exit 0
