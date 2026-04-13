#!/bin/bash
# Claude Code 代理启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HTTP_PORT="${PROXY_PORT:-4000}"
HOST="${PROXY_HOST:-127.0.0.1}"

echo "=== Claude Code Proxy Launcher ==="

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/.env"
    set +a
    echo "[✓] Loaded env from $SCRIPT_DIR/.env"
fi

if [ -z "${OPENCODE_API_KEY:-}" ]; then
    echo "[ERROR] Missing OPENCODE_API_KEY"
    echo "[HINT] Create $SCRIPT_DIR/.env with: OPENCODE_API_KEY=your-key"
    exit 1
fi

# 4. 检查端口占用
kill_port_listener() {
    local port="$1"
    local label="$2"
    local pid

    if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        return 0
    fi

    echo "[!] $label port $port is already in use, attempting to stop the listener..."
    lsof -nP -iTCP:"$port" -sTCP:LISTEN || true

    pid="$(lsof -tiTCP:"$port" -sTCP:LISTEN | head -n 1)"
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || sudo kill "$pid" 2>/dev/null || true
        sleep 1

        if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "    Listener still active, forcing kill on PID $pid..."
            kill -9 "$pid" 2>/dev/null || sudo kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi

    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "[ERROR] Failed to free $label port $port"
        lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
        echo "[HINT] Try: sudo lsof -nP -iTCP:$port -sTCP:LISTEN"
        exit 1
    fi

    echo "    Freed $label port $port"
}

kill_port_listener "$HTTP_PORT" "HTTP"

# 5. 启动代理
echo ""
echo "[+] Starting proxy (HTTP:$HTTP_PORT)..."
echo "    Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
if [ "$HTTP_PORT" -lt 1024 ]; then
    sudo env PROXY_HOST="$HOST" PROXY_PORT="$HTTP_PORT" python3 reasoning_proxy.py
else
    env PROXY_HOST="$HOST" PROXY_PORT="$HTTP_PORT" python3 reasoning_proxy.py
fi
