.PHONY: trade backtest analysis

trade:
	python trade.py

backtest:
	python backtest.py

analysis:
	shiny run --app-dir analysis app:app
