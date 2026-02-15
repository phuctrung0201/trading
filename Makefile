.PHONY: trade backtest monitor

trade:
	python trade.py

backtest:
	python backtest.py

monitor:
	cd influxdb && docker build -t trading-influxdb .
	-docker rm -f influxdb 2>/dev/null
	docker run -d -p 8086:8086 --name influxdb trading-influxdb
