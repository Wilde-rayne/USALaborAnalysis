#!/usr/bin/env bash
set -euo pipefail

echo "=== Prairie Insights Deploy Script ==="

# ——————————————————————————————
# Settings
# ——————————————————————————————
COMPOSE_FILE="docker-compose.yml"
OLLAMA_SERVICE_NAME="ollama"
DASH_SERVICE_NAME="dashboard"
OLLAMA_PORT=11434
DASH_PORT=8050
OLLAMA_MODELS=("llama2:chat")
FULL_REBUILD=false
DEV_MODE=false

case "${1:-}" in
  --full) FULL_REBUILD=true ;;
  --dev)  DEV_MODE=true  ;;
esac

# ——————————————————————————————
# Functions
# ——————————————————————————————
install_docker() {
  echo "-- Docker not found: installing"
  case "$OS" in
    linux)
      sudo apt-get update && sudo apt-get install -y docker.io
      sudo systemctl enable docker && sudo systemctl start docker
      ;;
    macos)
      command -v brew >/dev/null || { echo "Please install Homebrew: https://brew.sh"; exit 1; }
      brew install --cask docker
      echo "→ Start Docker Desktop manually."
      ;;
    windows)
      command -v choco >/dev/null || { echo "Please install Chocolatey: https://chocolatey.org"; exit 1; }
      choco install docker-desktop -y
      echo "→ Log out/in and start Docker Desktop."
      ;;
    *)
      echo "Unsupported OS; install Docker manually." >&2
      exit 1
      ;;
  esac
}

install_compose() {
  echo "-- Docker Compose not found: installing"
  case "$OS" in
    linux)
      sudo apt-get update && sudo apt-get install -y docker-compose-plugin
      ;;
    macos)
      brew install docker-compose
      ;;
    windows)
      choco install docker-compose -y
      ;;
    *)
      echo "Unsupported OS; install Docker Compose manually." >&2
      exit 1
      ;;
  esac
}

wait_for_model() {
  local model=$1
  echo -n "-- Waiting for model '$model'"
  until docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE_NAME" \
        curl -sf http://localhost:$OLLAMA_PORT/api/tags | grep -q "$model"; do
    echo -n "."
    sleep 2
  done
  echo " → $model ready"
}

# ——————————————————————————————
# Pre-checks
# ——————————————————————————————
if pgrep -f "ollama serve" >/dev/null; then
  echo "-- stopping host Ollama"
  sudo pkill -f "ollama serve" || true
  sleep 1
fi

for svc in "$OLLAMA_SERVICE_NAME" "$DASH_SERVICE_NAME"; do
  if docker compose -f "$COMPOSE_FILE" ps -q "$svc" | grep -q .; then
    echo "-- removing stale container: $svc"
    docker compose -f "$COMPOSE_FILE" rm -sf "$svc" || true
  fi
done

# OS detection
OS_TYPE=$(uname -s)
case "$OS_TYPE" in
  Linux*)   OS=linux  ;;
  Darwin*)  OS=macos  ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *)        OS=unknown ;;
esac
echo "-- detected OS: $OS"

# Docker & Compose install if needed
if ! command -v docker >/dev/null; then
  install_docker
else
  echo "-- docker found: $(docker --version)"
fi

if ! docker compose version >/dev/null && ! command -v docker-compose >/dev/null; then
  install_compose
else
  echo "-- docker compose available"
fi

# ——————————————————————————————
# Deploy
# ——————————————————————————————
if [ "$DEV_MODE" = true ]; then
  echo "=== Dev mode: hot-reloading dashboard only ==="
  docker compose -f "$COMPOSE_FILE" up -d --no-deps "$DASH_SERVICE_NAME"
  exit 0
fi

echo "=== Stopping existing containers ==="
docker compose -f "$COMPOSE_FILE" down --remove-orphans || true

echo "=== Building images ==="
if [ "$FULL_REBUILD" = true ]; then
  docker compose -f "$COMPOSE_FILE" build --no-cache
else
  docker compose -f "$COMPOSE_FILE" build --no-cache "$DASH_SERVICE_NAME"
fi

echo "=== Starting services ==="
docker compose -f "$COMPOSE_FILE" up -d

echo "=== Pulling required Ollama models ==="
for model in "${OLLAMA_MODELS[@]}"; do
  echo "-- pulling model: $model"
  docker compose -f "$COMPOSE_FILE" exec -T "$OLLAMA_SERVICE_NAME" ollama pull "$model" \
    || echo "-- model '$model' may already exist"
done

echo "=== Waiting for Ollama API ==="
until curl -sSf "http://localhost:$OLLAMA_PORT/api/version" >/dev/null; do
  echo -n "."
  sleep 1
done
echo " → Ollama API ready"

echo "=== Verifying Ollama models ==="
for model in "${OLLAMA_MODELS[@]}"; do
  wait_for_model "$model"
done

echo "=== Checking container health ==="
OLLAMA_CID=$(docker compose -f "$COMPOSE_FILE" ps -q "$OLLAMA_SERVICE_NAME")
DASH_CID=$(docker compose -f "$COMPOSE_FILE" ps -q "$DASH_SERVICE_NAME")
for cid in "$OLLAMA_CID" "$DASH_CID"; do
  status=$(docker inspect "$cid" --format '{{.State.Health.Status}}' || echo "none")
  if [ "$status" != "healthy" ]; then
    echo "✖ Container $cid is $status"
    docker logs "$cid"
    exit 1
  fi
  echo "✔ Container $cid is healthy"
done

echo "✅ All services are up!"
echo "  Dashboard: http://localhost:$DASH_PORT"
echo "  Ollama:    http://localhost:$OLLAMA_PORT"

# Open browser
case "$OS" in
  macos)   open "http://localhost:$DASH_PORT" ;;
  linux)   xdg-open "http://localhost:$DASH_PORT" || true ;;
  windows) start "http://localhost:$DASH_PORT" ;;
esac