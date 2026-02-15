.PHONY: trade backtest monitor monitor-down monitor-logs

trade:
	python trade.py

backtest:
	python backtest.py

monitor:
	-docker rm -f influxdb grafana 2>/dev/null
	docker-compose up -d

monitor-down:
	docker-compose down

monitor-logs:
	docker-compose logs -f
