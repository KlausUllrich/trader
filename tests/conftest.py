import numpy as np
import pandas as pd
import pytest

from trader.config import load_config


@pytest.fixture
def cfg():
    """The example config, but voting on the 1h bars themselves (short synthetic series)."""
    c = load_config("config.example.yaml")
    c["signal"].pop("timeframe", None)
    c["exits"].pop("max_days", None)
    return c


@pytest.fixture
def daily_cfg():
    """The example config as shipped: votes on daily bars."""
    return load_config("config.example.yaml")


def make_bars(n=3000, drift=0.0, vol=0.004, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100], close[:-1]])
    spread = np.abs(rng.standard_normal(n)) * vol * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


@pytest.fixture
def bars():
    return make_bars
