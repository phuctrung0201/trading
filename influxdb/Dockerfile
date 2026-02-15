FROM influxdb:2.7

COPY backtest-dashboard.yaml /docker-entrypoint-initdb.d/
COPY setup-dashboard.sh /docker-entrypoint-initdb.d/

ENV DOCKER_INFLUXDB_INIT_MODE=setup \
    DOCKER_INFLUXDB_INIT_USERNAME=admin \
    DOCKER_INFLUXDB_INIT_PASSWORD=admin123456 \
    DOCKER_INFLUXDB_INIT_ORG=trading \
    DOCKER_INFLUXDB_INIT_BUCKET=trading \
    DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=trading-super-secret-token

EXPOSE 8086
