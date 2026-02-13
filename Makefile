.PHONY: trade backtest monitor

trade:
	python trade.py

backtest:
	python backtest.py

monitor:
	docker-compose up -d influxdb monitor
	./monitor/provision_dashboard.sh
