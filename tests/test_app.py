import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from trader import app, data, levels, volatility


def test_volatility_forecast_tracks_regimes(bars):
    calm = bars(24 * 300, vol=0.002, seed=1)
    wild = bars(24 * 100, vol=0.008, seed=2, start=calm.index[-1] + pd.Timedelta(hours=1))
    wild[["open", "high", "low", "close"]] *= calm["close"].iloc[-1] / wild["open"].iloc[0]
    daily = data.resample(pd.concat([calm, wild]), "1D")
    fc = volatility.forecast(daily)
    assert fc["forecast"].iloc[-1] > 2 * fc["forecast"].loc[: calm.index[-1]].dropna().median()
    now = volatility.latest(daily)
    assert now["regime"] == "jumpy" and now["move"] > 0


def test_levels_split_around_price(bars):
    daily = data.resample(bars(24 * 400, seed=4), "1D")
    close = daily["close"].iloc[-1]
    found = levels.find(daily)
    assert found
    for x in found:
        assert (x["price"] > close) == (x["kind"] == "resistance")
        assert x["touches"] >= 1


def test_app_serves_all_screens(daily_cfg):
    c = TestClient(app.create(daily_cfg, demo=True))
    assert c.get("/").status_code == 200
    assert c.get("/static/vendor/lightweight-charts.standalone.production.js").status_code == 200

    rows = c.get("/api/overview").json()["rows"]
    assert [r["name"] for r in rows] == [i["name"] for i in daily_cfg["instruments"]]
    assert all("error" not in r for r in rows)

    for tf in ("1d", "1h"):
        ch = c.get(f"/api/chart/BTCUSD?tf={tf}").json()
        times = [x["time"] for x in ch["candles"]]
        assert times == sorted(times) and len(set(times)) == len(times)
        assert set(m["time"] for m in ch["markers"]) <= set(times)  # markers sit on candles

    chk = c.get("/api/check/BTCUSD?direction=short").json()
    p = chk["plan"]
    assert p["stop"] > p["entry"] > p["target"]
    assert np.isclose(p["risk"], daily_cfg["account"]["capital"] * daily_cfg["account"]["risk_per_trade"])
    assert any(x["name"] == "Whales" for x in chk["checks"])

    assert c.get("/api/chart/NOPE").status_code == 404
    assert c.get("/api/check/GOLD?direction=up").status_code == 400
