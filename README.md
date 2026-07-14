# Allied Health AI Platform

Research MVP for generating verified agentic mind maps from the `allied_health_edu`
PostgreSQL database, with an educator-facing Teaching Intelligence UI.

## One-command Docker stack

This starts **API**, **Streamlit frontend**, **Qdrant**, and **Redis** together.
PostgreSQL stays on the host (your existing `allied_health_edu` database).

```bash
cd allied_health_ai_platform
cp -n .env.example .env   # if needed, then set OPENAI_API_KEY

# Recommended helper (loads project-root .env correctly)
chmod +x docker/compose.sh
./docker/compose.sh up --build

# Equivalent one-liner
# docker compose --project-directory . --env-file .env -f docker/docker-compose.yml up --build
```

Then open:

- Frontend: http://localhost:8501
- API health: http://localhost:8000/api/v1/health
- API docs: http://localhost:8000/docs

Stop everything:

```bash
./docker/compose.sh down
```

### Postgres from Docker

Containers reach host Postgres via `host.docker.internal`.
Set this in `.env` if your credentials differ:

```bash
DOCKER_DATABASE_URL=postgresql://USER@host.docker.internal:5432/allied_health_edu
```

If the API cannot connect, ensure Postgres listens on TCP and allows Docker host connections
in `pg_hba.conf` (for local trust/password auth as appropriate).

### Indexed vector store

Qdrant data is stored in the Docker volume `qdrant_storage`.
After a fresh volume, re-index collections once:

```bash
for collection in papers resources programs communities simulation_cases; do
  curl -s -X POST http://localhost:8000/api/v1/index \
    -H "Content-Type: application/json" \
    -d "{\"collection\":\"$collection\",\"mode\":\"full\",\"batch_size\":40}"
  echo
done
```

## Local development (without app containers)

Use Python 3.11 or 3.12. A Conda env is recommended.

```bash
conda create -y -n allied_health_ai python=3.12 pip
conda activate allied_health_ai
cd allied_health_ai_platform
python -m pip install -e ".[dev]"
cp -n .env.example .env
```

Start only infra:

```bash
./docker/compose.sh up -d qdrant redis
```

Run app processes on the host:

```bash
uvicorn backend.app.main:app --reload --port 8000
API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py --server.port 8501
```

## Example planning question

```text
What interprofessional education resources address opioid education in rural Georgia counties?
```
