#!/bin/bash
# During init, InfluxDB runs on port 9999
influx apply -f /docker-entrypoint-initdb.d/backtest-dashboard.yaml \
  --host http://localhost:9999 \
  --org trading \
  --token trading-super-secret-token \
  --force yes
