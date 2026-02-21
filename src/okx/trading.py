from src.okx.auth import OkxAuth


class OkxTrading:
    def __init__(self, auth: OkxAuth):
        self._auth = auth

    def place_order(self, instrument: str, side: str, size: float,
                    order_type: str = "market") -> list:
        body = {
            "instId": instrument,
            "tdMode": "cross",
            "side": side,
            "ordType": order_type,
            "sz": str(size),
        }
        return self._auth.signed_request("POST", "/api/v5/trade/order", body)

    def get_order(self, instrument: str, order_id: str) -> dict:
        path = f"/api/v5/trade/order?instId={instrument}&ordId={order_id}"
        data = self._auth.signed_request("GET", path)
        if not data:
            return {}
        return data[0]

    def close_position(self, instrument: str):
        body = {"instId": instrument, "mgnMode": "cross"}
        return self._auth.signed_request("POST", "/api/v5/trade/close-position", body)
