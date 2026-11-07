#!/usr/bin/env bash
set -euo pipefail

# NOTE: The tessella-tracker binary has been replaced by the `musivum` binary
# from the musivum/ workspace. Update LOCAL_REPO and the binary paths accordingly
# when re-enabling this script.

### CONFIGURATION ###
LOCAL_REPO="${HOME}/Proyectos/musivum-tessella"
REMOTE_USER="vitichenko"
REMOTE_HOST="34.171.244.142"
REMOTE_DIR="/home/${REMOTE_USER}/musivum"
REMOTE_BIN_DIR="${REMOTE_DIR}/bin"
REMOTE_CONFIG_DIR="${REMOTE_DIR}/config"

CLEAN_DB=0
if [[ "${1-}" == "--clean" ]]; then
    CLEAN_DB=1
    echo "⚠️ --clean flag detected. Remote database will be wiped."
fi

echo "===================================================="
echo "   🚢 Deploying Musivum Orchestrator to GCP VM"
echo "   Target: ${REMOTE_USER}@${REMOTE_HOST}"
echo "===================================================="

### 1) Compile binary locally ###
echo ">>> [1/6] Compiling musivum in release..."
cd "${LOCAL_REPO}/musivum"
cargo build -p musivum --release

### 2) Prepare remote environment ###
echo ">>> [2/6] Preparing remote environment and stopping musivum..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    mkdir -p '${REMOTE_BIN_DIR}'
    mkdir -p '${REMOTE_CONFIG_DIR}'
    echo 'Stopping existing processes...'
    pkill -x musivum || true
    sleep 1

    if [ \"${CLEAN_DB}\" = \"1\" ]; then
        echo '🧹 Wiping remote database...'
        rm -f '${REMOTE_DIR}'/musivum.db*
    fi
"

### 3) Upload Configuration ###
echo ">>> [3/6] Uploading musivum.toml configuration..."
scp "${LOCAL_REPO}/musivum/config/musivum.toml" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CONFIG_DIR}/musivum.toml"

### 4) Upload binary to the VM ###
echo ">>> [4/6] Uploading musivum binary..."
scp "${LOCAL_REPO}/musivum/target/release/musivum" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BIN_DIR}/musivum"

### 5) Set execution permissions ###
echo ">>> [5/6] Ensuring execution permissions..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "chmod +x '${REMOTE_BIN_DIR}/musivum'"

### 6) Restart musivum in the VM ###
echo ">>> [6/6] Restarting musivum service..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    cd '${REMOTE_DIR}'
    # We use nohup so the process survives SSH session closure
    nohup '${REMOTE_BIN_DIR}/musivum' \
        --config '${REMOTE_CONFIG_DIR}/musivum.toml' \
        > '${REMOTE_DIR}/musivum.log' 2>&1 &
    
    sleep 3
    if pgrep -x musivum > /dev/null; then
        echo '✅ Musivum is RUNNING (PID: \$(pgrep -x musivum))'
    else
        echo '❌ FAILED to start Musivum. Last log lines:'
        tail -n 20 '${REMOTE_DIR}/musivum.log'
        exit 1
    fi
"

echo ""
echo "🚀 Deployment completed successfully."
echo "Status URL: http://${REMOTE_HOST}:8080/status"
echo "To see live logs:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ${REMOTE_DIR}/musivum.log'"
