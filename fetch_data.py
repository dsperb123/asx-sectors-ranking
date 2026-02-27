#!/usr/bin/env python3
"""
ASX 200 Sector Strength Ranking — data fetcher
Replicates the Pine Script bull/bear event detection, WMA weighting,
micro gap detection, composite scoring and rank trending.
Outputs: data/sectors.json
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config (mirrors Pine Script defaults) ─────────────────────────────────────
SECTORS = {
    "XBK.AX": "Banks",
    "XTJ.AX": "Comm Services",
    "XDJ.AX": "Cons Disc",
    "XSJ.AX": "Cons Staples",
    "XEJ.AX": "Energy",
    "XFJ.AX": "Financials",
    "XHJ.AX": "Health Care",
    "XNJ.AX": "Industrials",
    "XIJ.AX": "Info Tech",
    "XMJ.AX": "Materials",
    "XRE.AX": "Real Estate",
    "XJR.AX": "Resources",
    "XUJ.AX": "Utilities",
}

D_LOOKBACK       = 30
W_LOOKBACK       = 13
M_LOOKBACK       = 12
CL_LEN           = 10
CL_MULT          = 1.7
DAILY_WEIGHT     = 0.50
WEEKLY_WEIGHT    = 0.30
MONTHLY_WEIGHT   = 0.20
ENABLE_TICK_GAPS = True
WICK_THRESHOLD   = 1.0
ENABLE_MG        = True
MG_WEEKLY_LBK    = 13
RANK_LOOKBACK    = 10   # daily bars


# ── WMA (matches Pine Script ta.wma) ──────────────────────────────────────────
def wma(series: np.ndarray, period: int) -> np.ndarray:
    weights = np.arange(1, period + 1, dtype=float)
    w_sum   = weights.sum()
    out     = np.full(len(series), np.nan)
    for i in range(period - 1, len(series)):
        out[i] = np.dot(series[i - period + 1 : i + 1], weights) / w_sum
    return out


# ── Bull event detection (stateful, bar-by-bar) ───────────────────────────────
def detect_bull_events(opens, highs, lows, closes,
                       cl_len=CL_LEN, cl_mult=CL_MULT,
                       enable_tick_gaps=ENABLE_TICK_GAPS,
                       wick_threshold=WICK_THRESHOLD):
    n = len(closes)
    events = np.zeros(n, dtype=float)

    ranges = highs - lows
    # precompute range SMA (lagged by 1 — Pine Script uses rSma[1])
    range_sma = pd.Series(ranges).rolling(cl_len).mean().shift(1).values

    # stateful vars
    y_level, y_win = np.nan, 0
    in_hi,   in_win = np.nan, 0
    cl_mid,  cl_win, cl_ok = np.nan, 0, True
    tl_level, tl_win = np.nan, 0

    for i in range(5, n):
        bull = closes[i] > opens[i]

        # ── bull4
        bull4 = all(closes[i-k] > opens[i-k] for k in range(4))

        # ── open == low
        bull_ol = bull and opens[i] == lows[i]

        # ── outside / inside
        is_outside = highs[i] > highs[i-1] and lows[i] < lows[i-1]
        is_inside  = highs[i] < highs[i-1] and lows[i] > lows[i-1]

        # ── yellow (outside bar → wait for close above that bar's high)
        bull_yellow = False
        if is_outside and lows[i] < lows[i-1]:
            y_level, y_win = highs[i], 3
        elif y_win > 0:
            bull_yellow = closes[i] > y_level
            y_win -= 1
            if bull_yellow or y_win == 0:
                y_level, y_win = np.nan, 0

        # ── inside breakout (close above inside bar's high within 3 bars)
        bull_inside = False
        if is_inside:
            in_hi, in_win = highs[i], 3
        elif in_win > 0:
            bull_inside = closes[i] > in_hi
            in_win -= 1
            if bull_inside or in_win == 0:
                in_hi, in_win = np.nan, 0

        # ── climactic bull bar then holds above midpoint for 2 bars
        bull_clim = False
        r_sma = range_sma[i] if not np.isnan(range_sma[i]) else 0
        if bull and ranges[i] >= r_sma * cl_mult:
            cl_mid  = lows[i] + ranges[i] * 0.5
            cl_win  = 2
            cl_ok   = True
        elif cl_win > 0:
            cl_ok  = cl_ok and (lows[i] >= cl_mid)
            cl_win -= 1
            if cl_win == 0:
                bull_clim = True

        # ── bearish trending lows → close above prior high within 3 bars
        bull_tl = False
        if i >= 5:
            btl = lows[i-1] < lows[i-2] < lows[i-3] < lows[i-4]
            if btl:
                tl_level, tl_win = highs[i-1], 3
        if tl_win > 0 and not (i >= 5 and lows[i-1] < lows[i-2] < lows[i-3] < lows[i-4]):
            bull_tl = highs[i] > tl_level
            tl_win -= 1
            if bull_tl or tl_win == 0:
                tl_level, tl_win = np.nan, 0

        # ── tick gaps (price-unit wick threshold)
        tick_gap_dn = False
        brk_gap_up  = False
        if enable_tick_gaps and i >= 1:
            tick_gap_dn = (opens[i] < closes[i-1] and bull
                           and (opens[i] - lows[i]) < wick_threshold)
            brk_gap_up  = (opens[i] > closes[i-1] and closes[i] > highs[i-1])

        events[i] = float(any([bull4, bull_ol, bull_yellow, bull_inside,
                                bull_clim, bull_tl, tick_gap_dn, brk_gap_up]))
    return events


# ── Bear event detection (mirror of bull) ─────────────────────────────────────
def detect_bear_events(opens, highs, lows, closes,
                       cl_len=CL_LEN, cl_mult=CL_MULT,
                       enable_tick_gaps=ENABLE_TICK_GAPS,
                       wick_threshold=WICK_THRESHOLD):
    n = len(closes)
    events = np.zeros(n, dtype=float)

    ranges = highs - lows
    range_sma = pd.Series(ranges).rolling(cl_len).mean().shift(1).values

    y_level, y_win = np.nan, 0
    in_lo,   in_win = np.nan, 0
    cl_mid,  cl_win, cl_ok = np.nan, 0, True
    th_level, th_win = np.nan, 0

    for i in range(5, n):
        bear = closes[i] < opens[i]

        bear4 = all(closes[i-k] < opens[i-k] for k in range(4))
        bear_oh = bear and opens[i] == highs[i]

        is_outside = highs[i] > highs[i-1] and lows[i] < lows[i-1]
        is_inside  = highs[i] < highs[i-1] and lows[i] > lows[i-1]

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
        if bear and ranges[i] >= r_sma * cl_mult:
            cl_mid = lows[i] + ranges[i] * 0.5
            cl_win = 2
            cl_ok  = True
        elif cl_win > 0:
            cl_ok  = cl_ok and (highs[i] <= cl_mid)
            cl_win -= 1
            if cl_win == 0:
                bear_clim = True

        bear_th = False
        if i >= 5:
            bth = highs[i-1] > highs[i-2] > highs[i-3] > highs[i-4]
            if bth:
                th_level, th_win = lows[i-1], 3
        if th_win > 0 and not (i >= 5 and highs[i-1] > highs[i-2] > highs[i-3] > highs[i-4]):
            bear_th = lows[i] < th_level
            th_win -= 1
            if bear_th or th_win == 0:
                th_level, th_win = np.nan, 0

        tick_gap_up  = False
        brk_gap_dn   = False
        if enable_tick_gaps and i >= 1:
            tick_gap_up = (opens[i] > closes[i-1] and bear
                           and (highs[i] - opens[i]) < wick_threshold)
            brk_gap_dn  = (opens[i] < closes[i-1] and closes[i] < lows[i-1])

        events[i] = float(any([bear4, bear_oh, bear_yellow, bear_inside,
                                bear_clim, bear_th, tick_gap_up, brk_gap_dn]))
    return events


# ── Micro gap detection (weekly bars) ─────────────────────────────────────────
def detect_micro_gaps(highs, lows, closes, opens):
    """Returns (bull_events, bear_events) arrays."""
    n = len(closes)
    bull_ev = np.zeros(n, dtype=float)
    bear_ev = np.zeros(n, dtype=float)

    for i in range(2, n):
        bull = closes[i] > opens[i]
        bear = closes[i] < opens[i]

        bad_bull = bear and bull and closes[i] < lows[i-1]  # bear after bull, close below prior low
        bad_bear = bull and bear and closes[i] > highs[i-1]

        # Micro gap up: gap over 2 bars ago, progressive higher lows
        mg_bull = (lows[i]   > highs[i-2] and
                   lows[i-1] >= lows[i-2] and
                   lows[i]   >= lows[i-1])
        mg_bear = (highs[i]   < lows[i-2] and
                   highs[i-1] <= highs[i-2] and
                   highs[i]   <= highs[i-1])

        bull_ev[i] = float(mg_bull and not bad_bull)
        bear_ev[i] = float(mg_bear and not bad_bear)

    return bull_ev, bear_ev


# ── Bull % via WMA ─────────────────────────────────────────────────────────────
def calc_bull_pct_wma(bull_ev, bear_ev, lookback):
    b_wma = wma(bull_ev, lookback)
    s_wma = wma(bear_ev, lookback)
    total = b_wma + s_wma
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(total > 0, (b_wma / total) * 100.0, 50.0)
    return pct


# ── Resample OHLCV ────────────────────────────────────────────────────────────
def resample_ohlcv(df, freq):
    rule = {'W': 'W-FRI', 'M': 'ME'}[freq]
    r = df.resample(rule).agg({
        'Open': 'first', 'High': 'max',
        'Low': 'min',   'Close': 'last',
        'Volume': 'sum'
    }).dropna(subset=['Close'])
    return r


# ── Safe float ────────────────────────────────────────────────────────────────
def sf(v, n=2):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, n)
    except Exception:
        return None


# ── Process one sector ────────────────────────────────────────────────────────
def process_sector(ticker, name):
    print(f"  {ticker} ({name})")
    t = yf.Ticker(ticker)
    daily = t.history(period="2y")
    if daily.empty or len(daily) < 30:
        print(f"    [WARN] insufficient data", file=sys.stderr)
        return None

    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    daily = daily.sort_index()

    # ── Daily
    do = daily['Open'].values
    dh = daily['High'].values
    dl = daily['Low'].values
    dc = daily['Close'].values

    d_bull = detect_bull_events(do, dh, dl, dc)
    d_bear = detect_bear_events(do, dh, dl, dc)
    d_pct  = calc_bull_pct_wma(d_bull, d_bear, D_LOOKBACK)

    # ── Weekly
    wkly = resample_ohlcv(daily, 'W')
    wo = wkly['Open'].values
    wh = wkly['High'].values
    wl = wkly['Low'].values
    wc = wkly['Close'].values
    w_bull = detect_bull_events(wo, wh, wl, wc)
    w_bear = detect_bear_events(wo, wh, wl, wc)
    w_pct  = calc_bull_pct_wma(w_bull, w_bear, W_LOOKBACK)

    # Micro gaps (on weekly bars)
    mg_bull_ev, mg_bear_ev = detect_micro_gaps(wh, wl, wc, wo)
    mg_bull_cnt = float(np.nansum(mg_bull_ev[-MG_WEEKLY_LBK:]))
    mg_bear_cnt = float(np.nansum(mg_bear_ev[-MG_WEEKLY_LBK:]))

    # ── Monthly
    mnth = resample_ohlcv(daily, 'M')
    mo = mnth['Open'].values
    mh = mnth['High'].values
    ml = mnth['Low'].values
    mc = mnth['Close'].values
    m_bull = detect_bull_events(mo, mh, ml, mc)
    m_bear = detect_bear_events(mo, mh, ml, mc)
    m_pct  = calc_bull_pct_wma(m_bull, m_bear, M_LOOKBACK)

    # ── Current values
    d_val = sf(d_pct[-1])
    w_val = sf(w_pct[-1])
    m_val = sf(m_pct[-1] if len(m_pct) > 0 else np.nan)

    comp = (
        (d_val or 50) * DAILY_WEIGHT +
        (w_val or 50) * WEEKLY_WEIGHT +
        (m_val or 50) * MONTHLY_WEIGHT
    )

    # ── Price & change
    price = sf(dc[-1])
    prev  = sf(dc[-2]) if len(dc) >= 2 else None
    chg   = sf(((dc[-1] / dc[-2]) - 1) * 100) if len(dc) >= 2 and dc[-2] != 0 else None

    # ── Historical composite (RANK_LOOKBACK daily bars ago)
    if len(d_pct) > RANK_LOOKBACK and len(w_pct) > 0 and len(m_pct) > 0:
        d_hist = d_pct[-(RANK_LOOKBACK + 1)]
        # Weekly/monthly don't shift much in 10 days — use last available
        w_hist = w_pct[-1]
        m_hist = m_pct[-1]
        hist_comp = (
            (sf(d_hist) or 50) * DAILY_WEIGHT +
            (sf(w_hist) or 50) * WEEKLY_WEIGHT +
            (sf(m_hist) or 50) * MONTHLY_WEIGHT
        )
    else:
        hist_comp = comp

    # ── Sparkline (last 10 daily closes, normalised 0–100)
    spark_raw = dc[-10:].tolist()
    lo, hi = min(spark_raw), max(spark_raw)
    rng = hi - lo if hi != lo else 1
    spark = [sf((v - lo) / rng * 100) for v in spark_raw]

    return {
        "ticker": ticker.replace(".AX", ""),
        "name":   name,
        "price":  price,
        "chg":    chg,
        "daily":  d_val,
        "weekly": w_val,
        "monthly": m_val,
        "composite": sf(comp),
        "hist_composite": sf(hist_comp),
        "mg_bull": int(mg_bull_cnt),
        "mg_bear": int(mg_bear_cnt),
        "spark":  spark,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)

    results = []
    for ticker, name in SECTORS.items():
        try:
            row = process_sector(ticker, name)
            if row:
                results.append(row)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}", file=sys.stderr)

    if not results:
        print("ERROR: No data fetched.", file=sys.stderr)
        sys.exit(1)

    # ── Rank by composite (current)
    results.sort(key=lambda x: x['composite'] or 0, reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    # ── Historical ranks
    hist_sorted = sorted(results, key=lambda x: x['hist_composite'] or 0, reverse=True)
    hist_rank_map = {r['ticker']: i + 1 for i, r in enumerate(hist_sorted)}
    for r in results:
        r['hist_rank'] = hist_rank_map.get(r['ticker'], r['rank'])
        r['rank_diff'] = r['rank'] - r['hist_rank']  # negative = moved up

    out = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rank_lookback_days": RANK_LOOKBACK,
        "sectors": results,
    }

    with open("data/sectors.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n✓ {len(results)} sectors written to data/sectors.json")
    print(f"  Top:    {results[0]['name']} — {results[0]['composite']:.1f}%")
    print(f"  Bottom: {results[-1]['name']} — {results[-1]['composite']:.1f}%")


if __name__ == "__main__":
    main()
