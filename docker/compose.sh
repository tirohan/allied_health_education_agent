#!/usr/bin/env bash
# Convenience wrapper so project-root `.env` is used for Compose interpolation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec docker compose --env-file "$ROOT/.env" -f docker/docker-compose.yml "$@"
