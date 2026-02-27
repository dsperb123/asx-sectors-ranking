#!/usr/bin/env python3
"""
ASX 200 Sector Strength Ranking — data fetcher
Tick gaps and micro gaps removed.
Outputs: data/sectors.json
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
SECTORS = {
    "^AXBK": "Banks",
    "^AXTJ": "Comm Services",
    "^AXDJ": "Cons Disc",
    "^AXSJ": "Cons Staples",
    "^AXEJ": "Energy",
    "^AXFJ": "Financials",
    "^AXHJ": "Health Care",
    "^AXNJ": "Industrials",
    "^AXIJ": "Info Tech",
    "^AXMJ": "Materials",
    "^AXPJ": "Real Estate",
    "^AXJR": "Resources",
    "^AXUJ": "Utilities",
}

D_LOOKBACK     = 30
W_LOOKBACK     = 13
M_LOOKBACK     = 12
CL_LEN         = 10
CL_MULT        = 1.7
DAILY_WEIGHT   = 0.50
WEEKLY_WEIGHT  = 0.30
MONTHLY_WEIGHT = 0.20
RANK_LOOKBACK  = 10


# ── WMA ────────────────────────────────────────────────────────────────────────
def wma(series: np.ndarray, period: int) -> np.ndarray:
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()
    out = np.full(len(series), np.nan)
    for i in range(period - 1, len(series)):
        out[i] = np.dot(series[i - period + 1:i + 1], weights) / w_sum
    return out


# ── Bull Event Detection ───────────────────────────────────────────────────────
def detect_bull_events(opens, highs, lows, closes):
    n = len(closes)
    events = np.zeros(n, dtype=float)

    ranges = highs - lows
    range_sma = pd.Series(ranges).rolling(CL_LEN).mean().shift(1).values

    y_level, y_win = np.nan, 0
    in_hi, in_win = np.nan, 0
    cl_mid, cl_win, cl_ok = np.nan, 0, True
    tl_level, tl_win = np.nan, 0

    for i in range(5, n):
        bull = closes[i] > opens[i]

        bull4 = all(closes[i-k] > opens[i-k] for k in range(4))
        bull_ol = bull and opens[i] == lows[i]

        is_outside = highs[i] > highs[i-1] and lows[i] < lows[i-1]
        is_inside = highs[i] < highs[i-1] and lows[i] > lows[i-1]

        bull_yellow = False
        if is_outside and lows[i] < lows[i-1]:
            y_level, y_win = highs[i], 3
        elif y_win > 0:
            bull_yellow = closes[i] > y_level
            y_win -= 1
            if bull_yellow or y_win == 0:
                y_level, y_win = np.nan, 0

        bull_inside = False
        if is_inside:
            in_hi, in_win = highs[i], 3
        elif in_win > 0:
            bull_inside = closes[i] > in_hi
            in_win -= 1
            if bull_inside or in_win == 0:
                in_hi, in_win = np.nan, 0

        bull_clim = False
        r_sma = range_sma[i] if not np.isnan(range_sma[i]) else 0
        if bull and ranges[i] >= r_sma * CL_MULT:
            cl_mid = lows[i] + ranges[i] * 0.5
            cl_win = 2
            cl_ok = True
        elif cl_win > 0:
            cl_ok = cl_ok and (lows[i] >= cl_mid)
            cl_win -= 1
            if cl_win == 0 and cl_ok:
                bull_clim = True

        bull_tl = False
        if lows[i-1] < lows[i-2] < lows[i-3] < lows[i-4]:
            tl_level, tl_win = highs[i-1], 3

        if tl_win > 0:
            bull_tl = highs[i] > tl_level
            tl_win -= 1
            if bull_tl or tl_win == 0:
                tl_level, tl_win = np.nan, 0

        events[i] = float(any([
            bull4, bull_ol, bull_yellow,
            bull_inside, bull_clim, bull_tl
        ]))

    return events


# ── Bear Event Detection ───────────────────────────────────────────────────────
def detect_bear_events(opens, highs, lows, closes):
    n = len(closes)
    events = np.zeros(n, dtype=float)

    ranges = highs - lows
    range_sma = pd.Series(ranges).rolling(CL_LEN).mean().shift(1).values

    y_level, y_win = np.nan, 0
    in_lo, in_win = np.nan, 0
    cl_mid, cl_win, cl_ok = np.nan, 0, True
    th_level, th_win = np.nan, 0

    for i in range(5, n):
        bear = closes[i] < opens[i]

        bear4 = all(closes[i-k] < opens[i-k] for k in range(4))
        bear_oh = bear and opens[i] == highs[i]

        is_outside = highs[i] > highs[i-1] and lows[i] < lows[i-1]
        is_inside = highs[i] < highs[i-1] and lows[i] > lows[i-1]

        bear_yellow = False
        if is_outside and highs[i] > highs[i-1]:
            y_level, y_win = lows[i], 3
        elif y_win > 0:
            bear_yellow = closes[i] < y_level
            y_win -= 1
            if bear_yellow or y_win == 0:
                y_level, y_win = np.nan, 0

        bear_inside = False
        if is_inside:
            in_lo, in_win = lows[i], 3
        elif in_win > 0:
            bear_inside = closes[i] < in_lo
            in_win -= 1
            if bear_inside or in_win == 0:
                in_lo, in_win = np.nan, 0

        bear_clim = False
        r_sma = range_sma[i] if not np.isnan(range_sma[i]) else 0
        if bear and ranges[i] >= r_sma * CL_MULT:
            cl_mid = lows[i] + ranges[i] * 0.5
            cl_win = 2
            cl_ok = True
        elif cl_win > 0:
            cl_ok = cl_ok and (highs[i] <= cl_mid)
            cl_win -= 1
            if cl_win == 0 and cl_ok:
                bear_clim = True

        bear_th = False
        if highs[i-1] > highs[i-2] > highs[i-3] > highs[i-4]:
            th_level, th_win = lows[i-1], 3

        if th_win > 0:
            bear_th = lows[i] < th_level
            th_win -= 1
            if bear_th or th_win == 0:
                th_level, th_win = np.nan, 0

        events[i] = float(any([
            bear4, bear_oh, bear_yellow,
            bear_inside, bear_clim, bear_th
        ]))

    return events
