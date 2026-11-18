"""Glint dynamics between two consecutive same-light frames (session-level EDA only, Phase P2).

Quantifies the droplet "merge / split / evaporate" idea: glints that persist, appear (births),
disappear (deaths), merge or split. These are NOT single-shot model features (they need a time
series of one piece).

Rules (``match_glints``):
- Merge: 2+ previous glints whose nearest current glint is the same glint ``j``, each within
  ``max_dist_px + r_eq(j)`` of it, and ``area(j) >= area_ratio x`` the largest of them.
- Split: the same rule with previous and current swapped.
- Persist: one-to-one Hungarian matching (``scipy.optimize.linear_sum_assignment``) on centroid
  distance among the remaining glints; pairs closer than ``max_dist_px`` persist.
- Births: current glints not explained above; deaths: previous glints not explained above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _centroids(df: pd.DataFrame) -> np.ndarray:
    return df[["centroid_y", "centroid_x"]].to_numpy(dtype=float).reshape(-1, 2)


def _grouped(
    src: pd.DataFrame, dst: pd.DataFrame, dist: np.ndarray, max_dist_px: float, ratio: float
) -> tuple[int, set[int], set[int]]:
    """Events where 2+ ``src`` glints collapse onto one larger ``dst`` glint.

    ``dist`` is (len(src), len(dst)). Returns (n_events, src indices used, dst indices used).
    """
    events, used_src, used_dst = 0, set(), set()
    if dist.size == 0:
        return events, used_src, used_dst
    nearest = dist.argmin(axis=1)
    src_area = src["area"].to_numpy(dtype=float)
    dst_area = dst["area"].to_numpy(dtype=float)
    for j in np.unique(nearest):
        reach = max_dist_px + np.sqrt(dst_area[j] / np.pi)
        members = np.flatnonzero((nearest == j) & (dist[:, j] <= reach))
        if len(members) >= 2 and dst_area[j] >= ratio * src_area[members].max():
            events += 1
            used_src.update(int(i) for i in members)
            used_dst.add(int(j))
    return events, used_src, used_dst


def match_glints(
    prev: pd.DataFrame, curr: pd.DataFrame, max_dist_px: float, area_ratio: float
) -> dict[str, int]:
    """Counts of ``births``, ``deaths``, ``merges``, ``splits``, ``persist`` between two frames.

    ``prev``/``curr`` are glint property tables (``GlintResult.props_df``: ``centroid_y``,
    ``centroid_x``, ``area`` in px at the same resolution). ``max_dist_px`` and ``area_ratio``
    come from ``glint.match_px`` and ``glint.merge_area_ratio``.
    """
    d = cdist(_centroids(prev), _centroids(curr))
    merges, m_prev, m_curr = _grouped(prev, curr, d, max_dist_px, area_ratio)
    splits, s_curr, s_prev = _grouped(curr, prev, d.T, max_dist_px, area_ratio)
    free_p = [i for i in range(len(prev)) if i not in m_prev | s_prev]
    free_c = [j for j in range(len(curr)) if j not in m_curr | s_curr]
    persist = 0
    if free_p and free_c:
        sub = d[np.ix_(free_p, free_c)]
        rows, cols = linear_sum_assignment(sub)
        ok = sub[rows, cols] <= max_dist_px
        persist = int(ok.sum())
    return {
        "births": len(free_c) - persist,
        "deaths": len(free_p) - persist,
        "merges": merges,
        "splits": splits,
        "persist": persist,
    }
