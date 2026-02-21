from src.okx.auth import OkxAuth


class OkxAccount:
    def __init__(self, auth: OkxAuth):
        self._auth = auth

    def get_asset(self, currency: str = "USDT") -> float:
        data = self._auth.signed_request("GET", f"/api/v5/account/balance?ccy={currency}")
        if not data:
            return 0.0
        details = data[0].get("details", [])
        for detail in details:
            if detail.get("ccy") == currency:
                return float(detail.get("eq", 0) or detail.get("availBal", 0))
        return 0.0

    def get_positions(self, instrument: str) -> list:
        path = f"/api/v5/account/positions?instId={instrument}"
        return self._auth.signed_request("GET", path)

    def set_leverage(self, instrument: str, leverage: int, mgn_mode: str = "cross"):
        body = {
            "instId": instrument,
            "lever": str(leverage),
            "mgnMode": mgn_mode,
        }
        return self._auth.signed_request("POST", "/api/v5/account/set-leverage", body)
