.PHONY: trade trade-stop trade-status trade-logs backtest monitor monitor-down monitor-logs

trade:
	@mkdir -p .supervisor
	@supervisord -c supervisord.conf 2>/dev/null || true
	supervisorctl -c supervisord.conf start trade

trade-stop:
	supervisorctl -c supervisord.conf stop trade
	supervisorctl -c supervisord.conf shutdown

trade-status:
	supervisorctl -c supervisord.conf status

trade-logs:
	tail -f .supervisor/trade.stdout.log .supervisor/trade.stderr.log

backtest:
	python backtest.py

monitor:
	-docker rm -f influxdb grafana 2>/dev/null
	docker-compose up -d

monitor-down:
	docker-compose down

monitor-logs:
	docker-compose logs -f
