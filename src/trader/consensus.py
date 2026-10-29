"""Combine strategy votes into signals.

A signal fires on the bar where agreement first reaches the threshold:
- `min_agree` or more strategies vote the same direction, and
- no strategy votes the opposite direction.
Size scales with how many agree (see `size_by_agreement`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine(votes: pd.DataFrame, min_agree: int = 2, size_by_agreement: dict | None = None) -> pd.DataFrame:
    size_by_agreement = {int(k): float(v) for k, v in (size_by_agreement or {3: 1.0, 2: 0.5}).items()}
    longs = (votes == 1).sum(axis=1)
    shorts = (votes == -1).sum(axis=1)

    direction = np.where((longs >= min_agree) & (shorts == 0), 1,
                         np.where((shorts >= min_agree) & (longs == 0), -1, 0))
    direction = pd.Series(direction, index=votes.index)
    agree = pd.Series(np.where(direction == 1, longs, np.where(direction == -1, shorts, 0)), index=votes.index)

    # Fire only when the state is new: direction changed, or agreement rose to a bigger size.
    prev_dir = direction.shift(fill_value=0)
    prev_agree = agree.shift(fill_value=0)
    fresh = (direction != 0) & ((direction != prev_dir) | (agree > prev_agree))

    size = agree.map(lambda n: size_by_agreement.get(int(n), 0.0))
    out = pd.DataFrame({"direction": direction, "agree": agree, "size": size, "fire": fresh & (size > 0)})
    out["agreeing"] = [
        ",".join(c for c in votes.columns if votes.at[i, c] == d) if d else ""
        for i, d in zip(votes.index, direction)
    ]
    return out
