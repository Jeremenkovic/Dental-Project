.PHONY: dev demo docker test-demo clean

ROOT := $(shell pwd)

# ── Local dev (no Docker) ─────────────────────────────────────────────────────
dev: _check-redis
	@echo "Starting backend + worker + frontend…"
	@mkdir -p $(ROOT)/data/scans
	@cd $(ROOT)/backend && \
	 DATABASE_URL="sqlite+aiosqlite:///$(ROOT)/data/dental.db" \
	 STORAGE_PATH="$(ROOT)/data/scans" \
	 REDIS_URL="redis://localhost:6379/0" \
	 python3 -m uvicorn main:app --reload --port 8000 &
	@cd $(ROOT)/worker && \
	 PYTHONPATH="$(ROOT)/worker:$$PYTHONPATH" \
	 DATABASE_URL="sqlite:///$(ROOT)/data/dental.db" \
	 STORAGE_PATH="$(ROOT)/data/scans" \
	 REDIS_URL="redis://localhost:6379/0" \
	 DEMO_MODE="$(DEMO_MODE)" \
	 GAUSSIAN_SPLATTING_DIR="$(GAUSSIAN_SPLATTING_DIR)" \
	 celery -A tasks worker --loglevel=info --concurrency=1 &
	@cd $(ROOT)/client && VITE_API_URL=http://localhost:8000 npm run dev

demo:
	@$(MAKE) dev DEMO_MODE=true

# ── Docker ────────────────────────────────────────────────────────────────────
docker:
	docker compose up --build

docker-demo:
	DEMO_MODE=true docker compose up --build

# ── Quick smoke test ──────────────────────────────────────────────────────────
test-demo:
	@echo "Creating demo scan via API…"
	@SCAN_ID=$$(curl -s -X POST http://localhost:8000/scans/demo | python3 -c "import sys,json; print(json.load(sys.stdin)['scan_id'])") && \
	echo "Scan ID: $$SCAN_ID" && \
	echo "Polling…" && \
	while true; do \
	  STATUS=$$(curl -s http://localhost:8000/scans/$$SCAN_ID); \
	  STAGE=$$(echo $$STATUS | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['stage'],d['progress'])"); \
	  echo "  $$STAGE"; \
	  DONE=$$(echo $$STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])"); \
	  if [ "$$DONE" = "done" ] || [ "$$DONE" = "error" ]; then break; fi; \
	  sleep 2; \
	done && \
	echo "Final: $$STATUS"

# ── Generate demo arch locally (no server needed) ─────────────────────────────
demo-arch:
	@mkdir -p ./data/demo
	@cd worker && python demo_mode.py ../data/demo
	@echo "Demo arch written to ./data/demo/mesh.obj"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf data/ client/dist client/dist-electron __pycache__ **/__pycache__
	find . -name "*.pyc" -delete

_check-redis:
	@redis-cli ping > /dev/null 2>&1 || (echo "Redis not running. Start with: redis-server" && exit 1)
