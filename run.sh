#!/usr/bin/env bash
# MARS-GIS unified runner / helper
# Usage: ./run.sh [command]
# If no command is supplied, an interactive menu is shown.

set -euo pipefail

# -------- Configuration & Colors --------
RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; CYAN="\033[36m"; BOLD="\033[1m"; RESET="\033[0m"
PROJECT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_MODULE="mars_gis.main:app"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"
FRONTEND_DIR="$PROJECT_ROOT_DIR/frontend"
PY_SRC_DIR="$PROJECT_ROOT_DIR/src"
PYTHON=${PYTHON:-python}
UVICORN_CMD=${UVICORN_CMD:-uvicorn}
NPM_CMD=${NPM_CMD:-npm}
DOCKER_COMPOSE_FILE=${DOCKER_COMPOSE_FILE:-docker-compose.yml}

# -------- Helper Functions --------
log() { printf "%b[%s]%b %s\n" "$BLUE" "MARS-GIS" "$RESET" "$1"; }
warn() { printf "%b[warn]%b %s\n" "$YELLOW" "$RESET" "$1"; }
err() { printf "%b[err ]%b %s\n" "$RED" "$RESET" "$1" >&2; }
ok() { printf "%b[ ok ]%b %s\n" "$GREEN" "$RESET" "$1"; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }; }

load_env() {
  local env_file=""
  if [[ -f "$PROJECT_ROOT_DIR/.env" ]]; then
    env_file="$PROJECT_ROOT_DIR/.env"
    ok "Loading .env"
  elif [[ -f "$PROJECT_ROOT_DIR/.env.example" ]]; then
    env_file="$PROJECT_ROOT_DIR/.env.example"
    warn ".env not found, using .env.example (create .env to customize)."
  else
    warn "No .env or .env.example present. Proceeding with defaults."
    return 0
  fi

  # Robust parser: supports quoted values, ignores comments & blank lines, preserves inline comments only inside quotes.
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim leading/trailing whitespace
    line="${line%%$'\r'}"  # strip CR if present
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      local key="${BASH_REMATCH[1]}" raw_val="${BASH_REMATCH[2]}" val
      # If value is quoted (single or double), keep as-is (without stripping inline comments)
      if [[ "$raw_val" =~ ^".*"$ || "$raw_val" =~ ^'.*'$ ]]; then
        val=${raw_val}
        # Remove surrounding quotes but preserve inner text
        val=${val:1:${#val}-2}
      else
        # Strip inline comments beginning with unquoted #
        val=${raw_val%%#*}
        # Trim whitespace
        val="${val%%[[:space:]]*}" || true
        # Re-trim whitespace edges
        val="${val##+([[:space:]])}"
        val="${val%%+([[:space:]])}"
      fi
      export "$key"="$val"
    fi
  done < "$env_file"
}

check_backend_deps() {
  if ! $PYTHON - <<'EOF' >/dev/null 2>&1
import importlib
for m in ["fastapi","uvicorn","pydantic"]:
    importlib.import_module(m)
EOF
  then
    warn "Backend dependencies missing. Attempting automatic install from requirements.txt..."
    if [[ -f "$PROJECT_ROOT_DIR/requirements.txt" ]]; then
      # Try direct install first; if PEP 668 error occurs, create venv
      if ! INSTALL_OUTPUT=$($PYTHON -m pip install -r "$PROJECT_ROOT_DIR/requirements.txt" --upgrade 2>&1); then
        if grep -q "externally managed" <<<"$INSTALL_OUTPUT"; then
          warn "System Python is externally managed (PEP 668). Creating isolated virtual environment .venv";
          if [[ ! -d "$PROJECT_ROOT_DIR/.venv" ]]; then
            $PYTHON -m venv "$PROJECT_ROOT_DIR/.venv" || { err "Failed to create virtual env"; exit 1; }
          fi
          # shellcheck disable=SC1091
          source "$PROJECT_ROOT_DIR/.venv/bin/activate"
          PYTHON="$PROJECT_ROOT_DIR/.venv/bin/python"
          ok "Using virtual environment Python: $PYTHON"
          $PYTHON -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
          if [[ "${FULL_INSTALL:-false}" == "true" ]]; then
            ok "Installing full dependency set (requirements.txt) inside virtual environment"
            if $PYTHON -m pip install -r "$PROJECT_ROOT_DIR/requirements.txt" --upgrade; then
              ok "Full backend dependencies installed inside virtual environment"
            else
              err "Failed to install full dependencies inside virtual environment"; exit 1
            fi
          else
            ok "Installing minimal core backend deps (fastapi, uvicorn, pydantic) inside virtual environment (set FULL_INSTALL=true for full stack)"
            if $PYTHON -m pip install fastapi uvicorn[standard] pydantic --upgrade; then
              ok "Minimal core backend deps installed"
            else
              err "Failed to install minimal core deps"; exit 1
            fi
          fi
        else
          printf '%s\n' "$INSTALL_OUTPUT" >&2
          err "Automatic install failed (non PEP 668). Please run: pip install -r requirements.txt"
          exit 1
        fi
      else
        ok "Backend dependencies installed"
      fi
    else
      err "requirements.txt not found. Cannot auto-install backend dependencies."
      exit 1
    fi
  fi
  ok "Backend Python deps OK"
}

check_frontend_deps() {
  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    warn "frontend/node_modules not found. Installing..."
    (cd "$FRONTEND_DIR" && $NPM_CMD install)
  fi
  ok "Frontend dependencies ready"
}

start_backend() {
  load_env
  check_backend_deps
  export PYTHONPATH="$PY_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  ok "Starting backend at http://$BACKEND_HOST:$BACKEND_PORT (reload)"
  exec $UVICORN_CMD "$BACKEND_MODULE" --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload
}

start_frontend() {
  check_frontend_deps
  ok "Starting frontend dev server (PORT from create-react-app env if set)"
  cd "$FRONTEND_DIR"
  exec $NPM_CMD start
}

start_both() {
  load_env
  check_backend_deps
  check_frontend_deps
  export PYTHONPATH="$PY_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  ok "Launching backend & frontend (Ctrl+C to stop both)"
  FRONTEND_PID=""; BACKEND_PID=""
  ( cd "$FRONTEND_DIR" && $NPM_CMD start & ) || warn "Failed to launch frontend"
  FRONTEND_PID=$!
  # Give frontend a moment in case it exits immediately due to port in use
  sleep 2
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    warn "Frontend process exited early (possibly port in use). Continuing with backend only."
    FRONTEND_PID=""
  fi
  ( $UVICORN_CMD "$BACKEND_MODULE" --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload & ) || err "Failed to launch backend"
  BACKEND_PID=$!
  trap 'echo; warn "Stopping..."; [[ -n "$BACKEND_PID" ]] && kill $BACKEND_PID 2>/dev/null || true; [[ -n "$FRONTEND_PID" ]] && kill $FRONTEND_PID 2>/dev/null || true' INT TERM
  if [[ -n "$BACKEND_PID" ]]; then
    if [[ -n "$FRONTEND_PID" ]]; then
      wait $BACKEND_PID $FRONTEND_PID
    else
      wait $BACKEND_PID
    fi
  fi
}

run_tests() {
  load_env
  check_backend_deps
  if ! command -v pytest >/dev/null 2>&1; then
    warn "pytest not found, installing transiently..."; $PYTHON -m pip install --quiet pytest pytest-asyncio
  fi
  export PYTHONPATH="$PY_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  ok "Running backend tests"
  pytest -q || { err "Tests failed"; exit 1; }
  ok "Tests passed"
}

lint_backend() {
  load_env
  missing=()
  for c in flake8 mypy black isort; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
  if ((${#missing[@]})); then
    warn "Installing missing linters: ${missing[*]}"; $PYTHON -m pip install --quiet ${missing[*]}
  fi
  export PYTHONPATH="$PY_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  ok "Flake8"; flake8 src || true
  ok "Mypy"; mypy src || true
  ok "Black (check)"; black --check src || true
  ok "Isort (check)"; isort --check-only src || true
  ok "Linting completed (see above for any issues)"
}

format_backend() {
  load_env
  export PYTHONPATH="$PY_SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
  black src && isort src
  ok "Code formatted"
}

docker_up() {
  require_cmd docker
  require_cmd docker compose || true
  ok "Starting Docker services"
  docker compose -f "$DOCKER_COMPOSE_FILE" up -d
  docker compose ps
}

docker_down() {
  require_cmd docker
  ok "Stopping Docker services"
  docker compose -f "$DOCKER_COMPOSE_FILE" down
}

usage() {
  cat <<EOF
${BOLD}MARS-GIS Run Helper${RESET}

Usage: ./run.sh <command>

Common commands:
  backend        Start FastAPI backend (reload)
  frontend       Start React frontend (CRA dev server)
  both           Start backend & frontend together
  test           Run backend tests (pytest)
  lint           Run static analysis (flake8, mypy, black --check, isort --check)
  format         Auto-format backend code (black + isort)
  docker-up      Start full Docker stack (docker compose up -d)
  docker-down    Stop Docker stack
  help           Show this help

Environment variables:
  PYTHON            Python interpreter (default: python)
  UVICORN_CMD       Uvicorn command (default: uvicorn)
  NPM_CMD           Node package manager command (default: npm)
  BACKEND_HOST      Backend host (default: 0.0.0.0)
  BACKEND_PORT      Backend port (default: 8000)

Examples:
  ./run.sh backend
  ./run.sh frontend
  ./run.sh both
  BACKEND_PORT=9000 ./run.sh backend
EOF
}

interactive_menu() {
  echo -e "${BOLD}MARS-GIS Interactive Menu${RESET}"
  select opt in "backend" "frontend" "both" "test" "lint" "format" "docker-up" "docker-down" "help" "quit"; do
    case $opt in
      backend) start_backend; break ;;
      frontend) start_frontend; break ;;
      both) start_both; break ;;
      test) run_tests; break ;;
      lint) lint_backend; break ;;
      format) format_backend; break ;;
      docker-up) docker_up; break ;;
      docker-down) docker_down; break ;;
      help) usage; break ;;
      quit) exit 0 ;;
      *) echo "Invalid option" ;;
    esac
  done
}

main() {
  local cmd=${1:-}
  case "$cmd" in
    backend) shift; start_backend "$@" ;;
    frontend) shift; start_frontend "$@" ;;
    both) shift; start_both "$@" ;;
    test) shift; run_tests "$@" ;;
    lint) shift; lint_backend "$@" ;;
    format) shift; format_backend "$@" ;;
    docker-up) shift; docker_up "$@" ;;
    docker-down) shift; docker_down "$@" ;;
    help|-h|--help) usage ;;
    "")
      # Default: start full stack (backend + frontend)
      ok "No command provided – starting full stack (backend + frontend). Use ./run.sh help for options."
      start_both
      ;;
    *) err "Unknown command: $cmd"; usage; exit 1 ;;
  esac
}

main "$@"
