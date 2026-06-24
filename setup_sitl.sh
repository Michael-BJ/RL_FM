#!/bin/bash
# =============================================================================
# SETUP SCRIPT: Gazebo Harmonic + PX4 SITL untuk neo_planner
# Ubuntu 24.04 + ROS2 Jazzy
# Jalankan: bash ~/neo_planner_ws/setup_sitl.sh
# =============================================================================

set -e  # Stop jika ada error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${BLUE}[→]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo ""
echo "============================================================"
echo "  SETUP: Gazebo Harmonic + PX4 SITL"
echo "  Ubuntu 24.04 + ROS2 Jazzy"
echo "============================================================"
echo ""

# =============================================================================
# STEP 1: Gazebo Harmonic apt repository
# =============================================================================
info "STEP 1/5: Menambah Gazebo Harmonic apt repository..."

if [ ! -f /etc/apt/sources.list.d/gazebo-stable.list ] && \
   ! apt-cache show gz-harmonic &>/dev/null; then
    sudo apt-get install -y curl gnupg lsb-release
    sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        | sudo tee /etc/apt/sources.list.d/gazebo-stable.list
    sudo apt-get update -qq
    log "Gazebo repo berhasil ditambahkan"
else
    log "Gazebo repo sudah ada, skip"
fi

# =============================================================================
# STEP 2: Install Gazebo Harmonic + ROS-Gazebo bridge
# =============================================================================
info "STEP 2/5: Install Gazebo Harmonic + ros-gz packages..."

sudo apt-get install -y \
    gz-harmonic \
    ros-jazzy-ros-gz \
    ros-jazzy-gz-sim-vendor \
    ros-jazzy-gz-tools-vendor

log "Gazebo Harmonic dan ros-gz terinstall"

# Verifikasi gz binary
if ! command -v gz &>/dev/null; then
    warn "gz tidak ditemukan di PATH setelah install, mencoba source ulang..."
fi

# =============================================================================
# STEP 3: Clone PX4-Autopilot
# =============================================================================
info "STEP 3/5: Clone PX4-Autopilot..."

PX4_DIR="$HOME/PX4-Autopilot"

if [ -d "$PX4_DIR" ]; then
    warn "PX4-Autopilot sudah ada di $PX4_DIR, skip clone"
    cd "$PX4_DIR"
    git submodule update --init --recursive 2>/dev/null || true
else
    git clone https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR" --recursive --depth 1
    log "PX4-Autopilot berhasil di-clone"
fi

# =============================================================================
# STEP 4: Install PX4 dependencies
# =============================================================================
info "STEP 4/5: Install PX4 dependencies..."

cd "$PX4_DIR"

# Install via script resmi PX4 (non-interactive)
bash ./Tools/setup/ubuntu.sh --no-nuttx

log "PX4 dependencies terinstall"

# =============================================================================
# STEP 5: Build PX4 SITL
# =============================================================================
info "STEP 5/5: Build PX4 SITL (ini akan memakan waktu 20-40 menit)..."

cd "$PX4_DIR"

# Source ROS2 dulu agar build tools tersedia
source /opt/ros/jazzy/setup.bash

make px4_sitl_default

log "PX4 SITL berhasil di-build!"

# =============================================================================
# SELESAI - Verifikasi
# =============================================================================
echo ""
echo "============================================================"
echo "  VERIFIKASI"
echo "============================================================"

# Cek gz
if command -v gz &>/dev/null; then
    GZ_VER=$(gz --version 2>/dev/null | head -1)
    log "Gazebo: $GZ_VER"
else
    warn "gz tidak ditemukan di PATH — pastikan source /opt/ros/jazzy/setup.bash"
fi

# Cek PX4 binary
PX4_BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"
if [ -f "$PX4_BIN" ]; then
    log "PX4 binary: $PX4_BIN"
else
    err "PX4 binary tidak ditemukan di $PX4_BIN — build mungkin gagal"
fi

echo ""
echo "============================================================"
echo "  CARA MENJALANKAN"
echo "============================================================"
echo ""
echo "  source /opt/ros/jazzy/setup.bash"
echo "  source ~/neo_planner_ws/install/setup.bash"
echo "  ros2 launch neo_planner sitl.launch.py world:=poles"
echo ""
echo "============================================================"
echo "  SETUP SELESAI!"
echo "============================================================"
