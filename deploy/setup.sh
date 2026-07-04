#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Provisioning idempotente del Droplet para triaje-ia-tfg.
# Ejecutar como root desde el Droplet:
#   DUCKDNS_TOKEN=<token> bash deploy/setup.sh
# ----------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/triaje-ia-tfg}"
REPO_URL="${REPO_URL:-https://github.com/CARLOS/triaje-ia-tfg.git}"
MODEL="llama3.1:8b-instruct-q4_K_M"
DOMAIN="triaje.duckdns.org"

log()  { printf '\n\033[1;34m=> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || fail "Ejecuta como root (sudo bash deploy/setup.sh)"
[[ -n "${DUCKDNS_TOKEN:-}" ]] || fail "Falta DUCKDNS_TOKEN en el entorno"

log "1/14 — Swap 4 GB (red de seguridad anti-OOM)"
if ! swapon --show | grep -q swapfile; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Swap activada."
else
    echo "Swap ya activa, se omite."
fi

log "2/14 — Desactivar reboot automático de unattended-upgrades"
UNATTENDED=/etc/apt/apt.conf.d/50unattended-upgrades
if [[ -f "$UNATTENDED" ]]; then
    sed -i 's/Unattended-Upgrade::Automatic-Reboot "true"/Unattended-Upgrade::Automatic-Reboot "false"/' "$UNATTENDED"
    echo "Reboot automático desactivado."
fi

log "3/14 — Instalar Ollama"
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama ya instalado, se omite."
fi
systemctl enable --now ollama || true

log "4/14 — Descargar modelo $MODEL"
sleep 2
while ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
    echo "Esperando a Ollama..."
    sleep 3
done
if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    ollama pull "$MODEL"
else
    echo "Modelo ya descargado, se omite."
fi

log "5/14 — Instalar uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv ya instalado, se omite."
fi

log "6/14 — Clonar/actualizar repositorio"
if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

log "7/14 — Instalar dependencias del proyecto (uv sync)"
cd "$REPO_DIR"
uv sync

log "8/14 — Variables de entorno de producción"
install -m 600 "$REPO_DIR/deploy/.env.production" "$HOME/.env-production"

log "9/14 — DuckDNS"
mkdir -p /etc/duckdns
DUCKDNS_TOKEN="$DUCKDNS_TOKEN" envsubst < "$REPO_DIR/deploy/duckdns.sh.tpl" > /etc/duckdns/duck.sh
chmod +x /etc/duckdns/duck.sh
/etc/duckdns/duck.sh || true
( crontab -l 2>/dev/null | grep -v 'duck.sh' ; echo '*/5 * * * * /etc/duckdns/duck.sh >/dev/null 2>&1' ) | crontab -

log "10/14 — Instalar Caddy"
if ! command -v caddy >/dev/null 2>&1; then
    apt-get update
    apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
    apt-get update
    apt-get install -y caddy
else
    echo "Caddy ya instalado, se omite."
fi

log "11/14 — Caddyfile (basicauth)"
install -m 644 "$REPO_DIR/deploy/Caddyfile" /etc/caddy/Caddyfile
HASH_PLACEHOLDER='<PEGAR_HASH_AQUI>'
if grep -q "$HASH_PLACEHOLDER" /etc/caddy/Caddyfile; then
    echo ""
    echo "  ─────────────────────────────────────────────────────────"
    echo "  ATENCIÓN: Genera el hash de basicauth y pégalo en"
    echo "  /etc/caddy/Caddyfile:"
    echo "      caddy hash-password"
    echo "  (usuario sugerido: triaje)"
    echo "  Después ejecuta: systemctl reload caddy"
    echo "  ─────────────────────────────────────────────────────────"
fi

log "12/14 — Unidades systemd"
install -m 644 "$REPO_DIR/deploy/triaje.service"        /etc/systemd/system/triaje.service
install -m 644 "$REPO_DIR/deploy/ollama-warmup.service" /etc/systemd/system/ollama-warmup.service
# Ajustar ruta del ExecStart de uv si está en /root/.local/bin
if [[ -x /root/.local/bin/uv ]] && ! grep -q '/root/.local/bin/uv' /etc/systemd/system/triaje.service; then
    sed -i 's#ExecStart=.*uv run#ExecStart=/root/.local/bin/uv run#' /etc/systemd/system/triaje.service
fi

log "13/14 — Activar y arrancar servicios"
systemctl daemon-reload
systemctl enable --now ollama caddy triaje ollama-warmup

log "14/14 — Smoke test local"
sleep 3
for svc in ollama caddy triaje ollama-warmup; do
    printf "  %-18s %s\n" "$svc" "$(systemctl is-active "$svc")"
done
curl -sf http://127.0.0.1:11434/api/tags >/dev/null && echo "Ollama responde." || echo "Ollama NO responde (revisa journalctl -u ollama)."

cat <<EOF

============================================================
 DESPLIEGUE COMPLETADO
------------------------------------------------------------
 URL pública ......... https://$DOMAIN
 Repo en servidor .... $REPO_DIR
 Token DuckDNS ....... inyectado vía entorno (no versionado)
------------------------------------------------------------
 Si Caddy pidió hash: edita /etc/caddy/Caddyfile y
 ejecuta: systemctl reload caddy
 Para actualizar tras git push: bash deploy/update.sh
============================================================
EOF
