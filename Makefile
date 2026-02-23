.PHONY: migrate migrate-dry paper paper-stop paper-status paper-logs backtest backtest-all ingest monitor monitor-down monitor-logs

migrate:
	python -m src.migration.main

migrate-dry:
	python -m src.migration.main --dry-run

paper:
	@mkdir -p .supervisor
	@supervisord -c supervisord.conf 2>/dev/null || true
	supervisorctl -c supervisord.conf start paper

paper-one:
	python -m src.paper.main --setup $(setup)

paper-stop:
	supervisorctl -c supervisord.conf stop paper
	supervisorctl -c supervisord.conf shutdown

paper-status:
	supervisorctl -c supervisord.conf status

paper-logs:
	tail -f .supervisor/paper.stdout.log .supervisor/paper.stderr.log

backtest:
	python -m src.backtest.main --setup $(setup)

backtest-all:
	@for cfg in config/backtest/*.yaml; do \
		setup=$$(basename "$$cfg" .yaml); \
		echo "=== Backtest $$setup ==="; \
		python -m src.backtest.main --setup "$$setup"; \
	done

portfolio:
	python -m src.portfolio.main

ingest:
	python -m src.ingest.main --name $(name) --instrument $(instrument) --start $(start) --end $(end)

monitor:
	-docker rm -f clickhouse grafana 2>/dev/null
	docker-compose up -d

monitor-down:
	docker-compose down

monitor-logs:
	docker-compose logs -f
