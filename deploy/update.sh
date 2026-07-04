#!/usr/bin/env bash
# Actualiza el repo en el servidor y reinicia Streamlit.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/triaje-ia-tfg}"
cd "$REPO_DIR"

echo "=> git pull --ff-only"
git pull --ff-only

echo "=> uv sync"
uv sync

echo "=> systemctl restart triaje"
sudo systemctl restart triaje

echo "✓ Estado triaje: $(systemctl is-active triaje)"
echo "✓ Estado ollama: $(systemctl is-active ollama)"
echo "✓ Estado caddy:  $(systemctl is-active caddy)"
