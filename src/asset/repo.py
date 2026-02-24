from __future__ import annotations

from src.okx.auth import OkxAuth


class AssetRepo:
    def __init__(self, auth: OkxAuth):
        self._auth = auth

    def get_balance(self, currency: str = "USDT") -> float:
        data = self._auth.signed_request(
            "GET", f"/api/v5/account/balance?ccy={currency}",
        )
        if not data:
            return 0.0
        details = data[0].get("details", [])
        for detail in details:
            if detail.get("ccy") == currency:
                return float(detail.get("eq", 0) or detail.get("availBal", 0))
        return 0.0

    def get_positions(self, inst_id: str) -> list:
        return self._auth.signed_request(
            "GET", f"/api/v5/account/positions?instId={inst_id}",
        )

    def set_leverage(
        self, inst_id: str, leverage: int, mgn_mode: str = "cross",
    ) -> list:
        body = {
            "instId": inst_id,
            "lever": str(leverage),
            "mgnMode": mgn_mode,
        }
        return self._auth.signed_request(
            "POST", "/api/v5/account/set-leverage", body,
        )
