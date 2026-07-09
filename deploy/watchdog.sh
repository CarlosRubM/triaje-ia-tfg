#!/usr/bin/env bash
# Watchdog — reanuda procesos caídos de la app triaje en RunPod.
# Diseñado para correr desde cron cada minuto:
#   * * * * * /root/triaje-ia-tfg/deploy/watchdog.sh >>/var/log/watchdog.log 2>&1
set -uo pipefail
REPO_DIR=/root/triaje-ia-tfg
TUNNEL_ID="1e89744d-78cf-4bc7-994b-5025df3d877a"
TUNNEL_NAME="triaje-tfg"
TS=$(date '+%Y-%m-%dT%H:%M:%S')

log() { printf '%s %s\n' "$TS" "$*"; }

# 1. Ollama
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    log "Ollama caído — reiniciando"
    pkill -f 'ollama serve' 2>/dev/null || true
    sleep 1
    nohup ollama serve >/var/log/ollama.log 2>&1 &
    sleep 5
    # Pre-cargar el modelo en GPU
    curl -s http://127.0.0.1:11434/api/generate \
        -d '{"model":"llama3.1:8b-instruct-q4_K_M","keep_alive":-1}' >/dev/null 2>&1 || true
fi

# 2. Streamlit
if ! curl -sf http://127.0.0.1:8080 >/dev/null 2>&1; then
    log "Streamlit caído — reiniciando"
    pkill -f 'streamlit run' 2>/dev/null || true
    sleep 1
    cd "$REPO_DIR"
    OLLAMA_HOST=http://127.0.0.1:11434 nohup /usr/bin/uv run streamlit run src/triaje_ia/ui/app.py \
        --server.port=8080 --server.address=0.0.0.0 --server.headless=true \
        >/var/log/streamlit.log 2>&1 &
    sleep 6
fi

# 3. Cloudflared (named tunnel, público y permanente en triaje.me)
if ! pgrep -f "cloudflared tunnel run $TUNNEL_NAME" >/dev/null 2>&1; then
    log "Named tunnel caído — reiniciando"
    pkill -f 'cloudflared tunnel --url' 2>/dev/null || true
    pkill -f 'cloudflared tunnel run' 2>/dev/null || true
    sleep 1
    cd /root/.cloudflared
    nohup cloudflared tunnel run "$TUNNEL_NAME" >/var/log/cftunnel.log 2>&1 &
    sleep 10
fi

exit 0