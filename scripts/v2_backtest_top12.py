#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from market_data_sina import fetch_all_stocks, fetch_kline, is_main_board

ROOT = Path(__file__).resolve().parents[1]
V1_MODEL_PATH = ROOT / "config" / "model_v1.json"
OUT_JSON = ROOT / "reports" / "v2_top12_backtest.json"
OUT_CSV = ROOT / "reports" / "v2_top12_backtest.csv"
MODEL_OUT = ROOT / "config" / "model_v2.json"

LIMIT_UP = 0.095
MIN_HISTORY = 60
KLINE_COUNT = 220
WORKERS = 12

FULL_START = dt.date(2026, 3, 20)
DEV_START = dt.date(2026, 6, 1)
TRAIN_END = dt.date(2026, 7, 8)
OOS_START = dt.date(2026, 7, 9)
END = dt.date(2026, 9, 18)

# V2 is intentionally optimized for the user's actual execution capacity:
# only Top1 and Top2 matter. Hyperparameters are selected on the development
# window and then frozen before strict OOS evaluation.
TOPKS = (1, 2, 6)


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def pct(a, b):
    return (a / b - 1.0) * 100 if b else 0.0


def limit_up(bar, prev):
    pc = f(prev.get("close"))
    return pc > 0 and f(bar.get("close")) / pc - 1.0 >= LIMIT_UP


def zscore_clip(arr, mean, std):
    sd = max(float(std), 1e-9)
    return float(np.clip((arr - mean) / sd, -8.0, 8.0))


def stable_sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def model_score(row, model):
    z = f(model.get("intercept"))
    for item in model.get("features", []):
        sd = max(f(item.get("std"), 1.0), 1e-9)
        x = f(row.get(item.get("name")))
        z += f(item.get("coef")) * ((x - f(item.get("mean"))) / sd)
    return 1.0 / (1.0 + math.exp(max(-35.0, min(35.0, -z))))


def make_features(bars, i, ctx, v1_model):
    if i < MIN_HISTORY or i < 21:
        return None

    pre = bars[: i + 1]
    c = [f(x.get("close")) for x in pre]
    v = [f(x.get("volume")) for x in pre]
    t = [f(x.get("turnover")) for x in pre]
    a = [f(x.get("amount")) for x in pre]
    b, prev = bars[i], bars[i - 1]

    pc = f(prev.get("close"))
    o = f(b.get("open"))
    h = f(b.get("high"))
    l = f(b.get("low"))
    close = f(b.get("close"))
    if close <= 0 or pc <= 0:
        return None

    ma5 = statistics.fmean(c[-5:])
    ma10 = statistics.fmean(c[-10:])
    ma20 = statistics.fmean(c[-20:])
    ma60 = statistics.fmean(c[-60:])

    v5 = statistics.fmean(v[-5:])
    v20 = statistics.fmean(v[-20:])
    t5 = statistics.fmean(t[-5:])
    t20 = statistics.fmean(t[-20:])
    a5 = statistics.fmean(a[-5:])
    a20 = statistics.fmean(a[-20:])

    rng = max(h - l, 1e-9)
    open_gap = pct(o, pc)

    amps = [
        (f(x.get("high")) - f(x.get("low"))) / f(x.get("close")) * 100
        for x in pre[-20:] if f(x.get("close")) > 0
    ]
    avg_amp20 = statistics.fmean(amps) if amps else 1.0

    # Board-history features intentionally look only backward from T.
    zt_flags = [int(j > 0 and limit_up(pre[j], pre[j - 1])) for j in range(len(pre))]
    zt5 = sum(zt_flags[-5:])
    zt10 = sum(zt_flags[-10:])
    zt20 = sum(zt_flags[-20:])
    days_since_zt = 99
    for back in range(1, min(21, len(pre))):
        if zt_flags[-back]:
            days_since_zt = back
            break

    prev2 = bars[i - 2] if i >= 2 else {}
    prev2_close = f(prev2.get("close"))

    row = {
        # V1 32 features
        "ret_1d": pct(close, c[-2]),
        "ret_3d": pct(close, c[-4]),
        "ret_5d": pct(close, c[-6]),
        "ret_10d": pct(close, c[-11]),
        "ret_20d": pct(close, c[-21]),
        "volume_5_vs_20": v5 / v20 if v20 else 1.0,
        "close_above_ma20": int(close > ma20),
        "ma5_gt_ma10": int(ma5 > ma10),
        "ma10_gt_ma20": int(ma10 > ma20),
        "near_high20_pct": pct(close, max(c[-20:])),
        "above_low20_pct": pct(close, min(c[-20:])),
        "up_days_5": sum(c[z] > c[z - 1] for z in range(len(c) - 5, len(c))),
        "volatility_10d": statistics.pstdev(c[-10:]) / close * 100,
        "amplitude_1d": (h - l) / close * 100,
        "market_first_count": ctx["market_first_count"],
        "market_2plus_count": ctx["market_2plus_count"],
        "market_zt_count": ctx["market_zt_count"],
        "prev_market_first_count": ctx["prev_market_first_count"],
        "prev_market_2plus_count": ctx["prev_market_2plus_count"],
        "prev_market_1to2_rate": ctx["prev_market_1to2_rate"],
        "open_gap_pct": open_gap,
        "open_gap_band_high": int(open_gap >= 8.5),
        "open_gap_band_mid": int(5 <= open_gap < 8.5),
        "open_gap_band_low": int(open_gap < 5),
        "open_to_close_pct": pct(close, o),
        "lower_wick_ratio": (o - l) / rng,
        "intraday_pullback_pct": pct(o, l),
        "open_position_in_day": (o - l) / rng,
        "amplitude_vs_20d": ((h - l) / pc * 100) / avg_amp20 if pc else 0.0,
        "volume_vs_20d": f(b.get("volume")) / v20 if v20 else 1.0,
        "near_limit_open": int(open_gap >= 8.0),
        "close_near_high": int(h > 0 and (h - close) / h <= 0.002),

        # V2 structural additions
        "turnover_1d": t[-1],
        "turnover_5d_avg": t5,
        "turnover_20d_avg": t20,
        "turnover_5_vs_20": t5 / t20 if t20 else 1.0,
        "turnover_vs_20d": t[-1] / t20 if t20 else 1.0,
        "amount_5_vs_20": a5 / a20 if a20 else 1.0,
        "amount_vs_20d": a[-1] / a20 if a20 else 1.0,
        "price_vs_ma60_pct": pct(close, ma60),
        "ma20_vs_ma60_pct": pct(ma20, ma60),
        "prev_ret_1d": pct(pc, prev2_close) if prev2_close > 0 else 0.0,
        "prev_range_pct": (f(prev.get("high")) - f(prev.get("low"))) / pc * 100 if pc else 0.0,
        "prev_volume_vs_20d": f(prev.get("volume")) / v20 if v20 else 1.0,
        "prev_turnover_vs_20d": f(prev.get("turnover")) / t20 if t20 else 1.0,
        "preboard_5d_return": pct(pc, c[-6]),
        "preboard_5d_range_pct": pct(max(c[-6:-1]), min(c[-6:-1])) if len(c) >= 6 else 0.0,
        "board_history_5": zt5,
        "board_history_10": zt10,
        "board_history_20": zt20,
        "days_since_zt": min(days_since_zt, 30),
        "market_2plus_ratio": ctx["market_2plus_count"] / max(ctx["market_zt_count"], 1),
        "market_first_ratio": ctx["market_first_count"] / max(ctx["market_zt_count"], 1),
        "market_zt_vs_prev": ctx["market_zt_count"] / max(ctx["prev_market_first_count"] + ctx["prev_market_2plus_count"], 1),
        "close_position_day": (close - l) / rng,
        "upper_wick_ratio": (h - max(o, close)) / rng,
        "close_vs_open_pct": pct(close, o),
        "volume_price_confirm": (f(b.get("volume")) / v20 if v20 else 1.0) * max(pct(close, o), 0.0),
    }

    # Keep a diagnostic V1 score as an explicit stacking feature.
    row["v1_score"] = model_score(row, v1_model)
    return row


def prepare_bars(raw):
    bars = sorted(raw, key=lambda x: str(x.get("date", "")))
    idx = {str(x.get("date", "")): i for i, x in enumerate(bars)}
    return bars, idx


def build_market_states(stock_data):
    dates = sorted({
        dt.date.fromisoformat(str(bar.get("date")))
        for _, _, bars in stock_data.values()
        for bar in bars
        if bar.get("date")
    })
    states = {}
    for d in dates:
        ds = d.isoformat()
        first, two, zt = set(), set(), set()
        for code, (_, idx_map, bars) in stock_data.items():
            i = idx_map.get(ds)
            if i is None or i == 0:
                continue
            if not limit_up(bars[i], bars[i - 1]):
                continue
            zt.add(code)
            if i >= 2 and limit_up(bars[i - 1], bars[i - 2]):
                two.add(code)
            else:
                first.add(code)
        states[d] = {"first": first, "two": two, "zt": zt}
    return dates, states


def build_dataset(stock_data, market, dates, v1_model):
    rows = []
    daily = {}

    eligible_dates = [d for d in dates if FULL_START <= d <= END]
    for pos, d in enumerate(eligible_dates[:-1]):
        next_d = eligible_dates[pos + 1]
        cur = market.get(d, {"first": set(), "two": set(), "zt": set()})
        prev = market.get(eligible_dates[pos - 1], {"first": set(), "two": set(), "zt": set()}) if pos > 0 else cur
        prev_first = prev["first"]

        # IMPORTANT: use only information available at T when constructing X.
        prev_rate = (
            len(prev_first & cur["two"]) / len(prev_first)
            if prev_first else 0.16185320352130533
        )
        ctx = {
            "market_first_count": len(cur["first"]),
            "market_2plus_count": len(cur["two"]),
            "market_zt_count": len(cur["zt"]),
            "prev_market_first_count": len(prev_first),
            "prev_market_2plus_count": len(prev["two"]),
            "prev_market_1to2_rate": prev_rate,
        }

        day_rows = []
        for code, (name, idx_map, bars) in stock_data.items():
            i = idx_map.get(d.isoformat())
            j = idx_map.get(next_d.isoformat())
            if i is None or j != i + 1 or i < MIN_HISTORY:
                continue
            if not limit_up(bars[i], bars[i - 1]):
                continue
            if i >= 2 and limit_up(bars[i - 1], bars[i - 2]):
                continue

            feat = make_features(bars, i, ctx, v1_model)
            if feat is None:
                continue
            row = {
                "date": d.isoformat(),
                "prediction_date": next_d.isoformat(),
                "code": code,
                "name": name,
                "is_2board": int(limit_up(bars[j], bars[j - 1])),
            }
            row.update(feat)
            rows.append(row)
            day_rows.append(row)

        daily[d.isoformat()] = day_rows

    return rows, daily


def fit_logistic(X, y, l2=1.0, pos_weight=1.0, epochs=260, lr=0.05):
    # Standardized X is expected. Full-batch gradient descent is sufficient
    # for this compact research model and avoids a heavyweight ML dependency.
    n, p = X.shape
    w = np.zeros(p, dtype=np.float64)
    b = 0.0

    weights = np.where(y > 0.5, pos_weight, 1.0)
    weights = weights / np.mean(weights)

    for epoch in range(epochs):
        z = X @ w + b
        p_hat = stable_sigmoid(z)
        err = (p_hat - y) * weights
        grad_w = (X.T @ err) / n + l2 * w / n
        grad_b = float(np.mean(err))

        step = lr / math.sqrt(1.0 + epoch * 0.03)
        w -= step * grad_w
        b -= step * grad_b

    return w, float(b)


def rank_percentile(values):
    # Percentile rank with deterministic tie handling.
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    out = np.empty(len(arr), dtype=float)
    if len(arr) == 1:
        return np.ones(1, dtype=float)
    out[order] = np.arange(len(arr), dtype=float) / (len(arr) - 1)
    return out


def attach_scores(daily_rows, v2_w, v2_b, feature_names, means, stds, alpha):
    scored_days = {}
    for day, rs in daily_rows.items():
        if not rs:
            continue
        X = np.array([
            [zscore_clip(np.array([f(r.get(name))]), means[name], stds[name])[0] for name in feature_names]
            for r in rs
        ], dtype=float)
        v2_prob = stable_sigmoid(X @ v2_w + v2_b)
        v1 = np.array([f(r.get("v1_score")) for r in rs], dtype=float)
        v1_rank = rank_percentile(v1)
        v2_rank = rank_percentile(v2_prob)
        # Rank-level blend is deliberate: the task is daily Top1/Top2 selection,
        # not probability calibration across different market regimes.
        final = (1.0 - alpha) * v1_rank + alpha * v2_rank
        order = np.argsort(-final, kind="mergesort")
        day_out = []
        for rank_idx, ix in enumerate(order, 1):
            r = dict(rs[ix])
            r["v2_prob"] = float(v2_prob[ix])
            r["v1_rank"] = float(v1_rank[ix])
            r["v2_rank"] = float(v2_rank[ix])
            r["v2_score"] = float(final[ix])
            r["rank"] = rank_idx
            day_out.append(r)
        scored_days[day] = day_out
    return scored_days


def metrics(scored_days, start, end):
    selected = {k: 0 for k in TOPKS}
    hits = {k: 0 for k in TOPKS}
    days = {k: 0 for k in TOPKS}
    day_hit = {k: 0 for k in TOPKS}

    for day, rs in scored_days.items():
        d = dt.date.fromisoformat(day)
        if not (start <= d <= end) or not rs:
            continue
        for k in TOPKS:
            top = rs[:k]
            if len(top) < k:
                continue
            days[k] += 1
            selected[k] += k
            h = sum(int(x["is_2board"]) for x in top)
            hits[k] += h
            day_hit[k] += int(h > 0)

    out = {}
    for k in TOPKS:
        out[str(k)] = {
            "k": k,
            "days": days[k],
            "hits": hits[k],
            "selected_positions": selected[k],
            "precision": hits[k] / selected[k] if selected[k] else None,
            "daily_any_hit_rate": day_hit[k] / days[k] if days[k] else None,
        }
    return out


def objective(m):
    t1 = m.get("1", {}).get("precision") or 0.0
    t2 = m.get("2", {}).get("precision") or 0.0
    return 0.6 * t1 + 0.4 * t2


def main():
    v1_model = json.loads(V1_MODEL_PATH.read_text(encoding="utf-8"))

    print("Fetching main-board historical data from Sina...")
    quotes = [
        r for r in fetch_all_stocks()
        if is_main_board(str(r.get("code", "")), str(r.get("name", "")))
    ]
    print(f"Main-board universe: {len(quotes)}")

    stock_data = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        jobs = {ex.submit(fetch_kline, str(r["code"]), KLINE_COUNT): r for r in quotes}
        for fut in as_completed(jobs):
            meta = jobs[fut]
            try:
                bars = fut.result()
                if len(bars) < MIN_HISTORY + 1:
                    continue
                bars, idx = prepare_bars(bars)
                stock_data[str(meta["code"])] = (str(meta["name"]), idx, bars)
            except Exception:
                continue

    print(f"Usable stocks: {len(stock_data)}")
    dates, market = build_market_states(stock_data)
    rows, daily = build_dataset(stock_data, market, dates, v1_model)
    print(f"First-board events in window: {len(rows)}")

    feature_names = [
        x["name"] for x in v1_model["features"]
    ] + [
        "turnover_1d", "turnover_5d_avg", "turnover_20d_avg",
        "turnover_5_vs_20", "turnover_vs_20d", "amount_5_vs_20",
        "amount_vs_20d", "price_vs_ma60_pct", "ma20_vs_ma60_pct",
        "prev_ret_1d", "prev_range_pct", "prev_volume_vs_20d",
        "prev_turnover_vs_20d", "preboard_5d_return", "preboard_5d_range_pct",
        "board_history_5", "board_history_10", "board_history_20", "days_since_zt",
        "market_2plus_ratio", "market_first_ratio", "market_zt_vs_prev",
        "close_position_day", "upper_wick_ratio", "close_vs_open_pct",
        "volume_price_confirm", "v1_score"
    ]

    # Strict three-way design:
    #   1) fit period: 2026-03-20 .. 2026-05-31
    #   2) development validation: 2026-06-01 .. 2026-07-08
    #   3) strict OOS: 2026-07-09 .. 2026-09-18
    # No parameter is selected using strict OOS.
    fit_end = DEV_START - dt.timedelta(days=1)
    fit_rows = [
        r for r in rows
        if FULL_START <= dt.date.fromisoformat(r["date"]) <= fit_end
    ]
    dev_rows = [
        r for r in rows
        if DEV_START <= dt.date.fromisoformat(r["date"]) <= TRAIN_END
    ]
    pre_oos_rows = [
        r for r in rows
        if FULL_START <= dt.date.fromisoformat(r["date"]) <= TRAIN_END
    ]

    def standardize_params(rs):
        mu, sd = {}, {}
        for name in feature_names:
            vals = np.array([f(r.get(name)) for r in rs], dtype=float)
            mu[name] = float(np.mean(vals))
            std = float(np.std(vals))
            sd[name] = max(std, 1e-6)
        return mu, sd

    def matrix(rs, mu, sd):
        return np.array(
            [[zscore_clip(np.array([f(r.get(name))]), mu[name], sd[name])[0] for name in feature_names] for r in rs],
            dtype=float,
        )

    fit_means, fit_stds = standardize_params(fit_rows)
    X_fit0 = matrix(fit_rows, fit_means, fit_stds)
    y_fit0 = np.array([int(r["is_2board"]) for r in fit_rows], dtype=float)

    oos_count = sum(1 for r in rows if OOS_START <= dt.date.fromisoformat(r["date"]) <= END)
    print(
        f"Fit rows: {len(fit_rows)}; positives: {int(y_fit0.sum())}; "
        f"Dev rows: {len(dev_rows)}; OOS rows: {oos_count}"
    )

    configs = [
        (l2, pos_weight)
        for l2 in (0.25, 0.5, 1.0, 2.0)
        for pos_weight in (1.0, 1.5, 2.0)
    ]

    candidates = []
    # Each candidate is fitted ONLY on the pre-dev fit period.
    for l2, pos_weight in configs:
        w, b = fit_logistic(X_fit0, y_fit0, l2=l2, pos_weight=pos_weight)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            dev_daily = attach_scores(
                daily, w, b, feature_names, fit_means, fit_stds, alpha
            )
            mm = metrics(dev_daily, DEV_START, TRAIN_END)
            candidates.append({
                "l2": l2,
                "pos_weight": pos_weight,
                "alpha_v2": alpha,
                "objective": objective(mm),
                "dev_metrics": mm,
            })

    candidates.sort(
        key=lambda x: (
            -x["objective"],
            -((x["dev_metrics"]["1"]["precision"] or 0.0)),
            -((x["dev_metrics"]["2"]["precision"] or 0.0)),
            x["alpha_v2"],
        )
    )
    best = candidates[0]
    print("Best development configuration:", json.dumps(best, ensure_ascii=False, indent=2))

    # Final V2 model: refit using ALL data available before strict OOS starts.
    final_means, final_stds = standardize_params(pre_oos_rows)
    X_final = matrix(pre_oos_rows, final_means, final_stds)
    y_final = np.array([int(r["is_2board"]) for r in pre_oos_rows], dtype=float)
    best_w, best_b = fit_logistic(
        X_final,
        y_final,
        l2=best["l2"],
        pos_weight=best["pos_weight"],
    )

    all_scored = attach_scores(
        daily, best_w, best_b, feature_names, final_means, final_stds, best["alpha_v2"]
    )
    full_m = metrics(all_scored, FULL_START, END)
    strict_m = metrics(all_scored, OOS_START, END)
    dev_m = metrics(all_scored, DEV_START, TRAIN_END)

    # V1 reference metrics are embedded in the repo's historical result notes.
    # We also compute an exact same-universe V1 ranking in this run.
    v1_daily = {}
    for day, rs in daily.items():
        rr = sorted((dict(r) for r in rs), key=lambda x: (-f(x["v1_score"]), x["code"]))
        for rank, r in enumerate(rr, 1):
            r["rank"] = rank
        v1_daily[day] = rr
    v1_dev = metrics(v1_daily, DEV_START, TRAIN_END)
    v1_oos = metrics(v1_daily, OOS_START, END)
    v1_full = metrics(v1_daily, FULL_START, END)

    improvements = {
        "full": {
            str(k): (
                (full_m[str(k)]["precision"] - v1_full[str(k)]["precision"])
                if full_m[str(k)]["precision"] is not None and v1_full[str(k)]["precision"] is not None else None
            )
            for k in TOPKS
        },
        "strict_oos": {
            str(k): (
                (strict_m[str(k)]["precision"] - v1_oos[str(k)]["precision"])
                if strict_m[str(k)]["precision"] is not None and v1_oos[str(k)]["precision"] is not None else None
            )
            for k in TOPKS
        },
    }

    # Export a runnable V2 model configuration. V2 stays independent of V1.
    model_out = {
        "version": "one-to-two-v2-top12-rank-ensemble",
        "trained_through": TRAIN_END.isoformat(),
        "feature_count": len(feature_names),
        "features": [
            {"name": name, "mean": final_means[name], "std": final_stds[name], "coef": float(best_w[i])}
            for i, name in enumerate(feature_names)
        ],
        "intercept": float(best_b),
        "alpha_v2": float(best["alpha_v2"]),
        "selection_objective": "0.6*Top1_precision + 0.4*Top2_precision on development window",
        "hyperparameters": {
            "l2": best["l2"],
            "pos_weight": best["pos_weight"],
        },
        "strict_oos": {
            "start": OOS_START.isoformat(),
            "end": END.isoformat(),
            "metrics": strict_m,
            "v1_metrics_same_run": v1_oos,
        },
        "notes": [
            "Independent V2; does not modify config/model_v1.json.",
            "Main-board only: excludes STAR, ChiNext, Beijing, ST, *ST and delisting names.",
            "Uses daily Sina K-line data, same as V1.",
            "The ranking is optimized for daily Top1/Top2 precision rather than probability calibration.",
            "Hyperparameter selection is fitted on 2026-03-20 through 2026-05-31 and validated on 2026-06-01 through 2026-07-08; final refit then uses all pre-OOS data through 2026-07-08. Strict OOS starts 2026-07-09.",
            "Historical universe is based on currently retrievable main-board securities; survivorship bias remains possible.",
        ],
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps(model_out, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "version": model_out["version"],
        "data_source": "Sina",
        "universe": "沪深主板",
        "window": {"full_start": FULL_START.isoformat(), "end": END.isoformat()},
        "development": {
            "fit_start": FULL_START.isoformat(),
            "fit_end": fit_end.isoformat(),
            "refit_start": FULL_START.isoformat(),
            "refit_end": TRAIN_END.isoformat(),
            "validation_start": DEV_START.isoformat(),
            "validation_end": TRAIN_END.isoformat(),
            "best_config": best,
            "metrics": dev_m,
            "v1_same_run_metrics": v1_dev,
        },
        "strict_oos": {
            "start": OOS_START.isoformat(),
            "end": END.isoformat(),
            "metrics": strict_m,
            "v1_same_run_metrics": v1_oos,
        },
        "full_window": {
            "metrics": full_m,
            "v1_same_run_metrics": v1_full,
        },
        "improvements": improvements,
        "sample_counts": {
            "stocks": len(stock_data),
            "first_board_events": len(rows),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
        },
        "top12_focus": {
            "primary_metrics": ["Top1 precision", "Top2 precision", "Top2 combined position precision"],
            "note": "Top6 retained only as a secondary context metric; V2 selection objective does not optimize Top6.",
        },
        "experiments": candidates[:20],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "window", "prediction_date", "rank", "code", "name",
            "v1_score", "v2_score", "v2_prob", "is_2board"
        ])
        for label, dct, start, end in [
            ("dev", all_scored, DEV_START, TRAIN_END),
            ("strict_oos", all_scored, OOS_START, END),
        ]:
            for day, rs in dct.items():
                d = dt.date.fromisoformat(day)
                if not (start <= d <= end):
                    continue
                for r in rs[:6]:
                    w.writerow([
                        label, r["prediction_date"], r["rank"], r["code"],
                        r["name"], r["v1_score"], r["v2_score"],
                        r["v2_prob"], r["is_2board"]
                    ])

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
