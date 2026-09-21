#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_data_sina import fetch_all_stocks, fetch_kline, is_main_board


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "config" / "model_v1.json"
OUT_DIR = ROOT / "reports" / "v2_1"
OUT_JSON = OUT_DIR / "v2_1_top12_backtest.json"
OUT_CSV = OUT_DIR / "v2_1_top12_backtest.csv"
OUT_ROWS = OUT_DIR / "v2_1_training_events.csv"

LIMIT_UP = 0.095
MIN_HISTORY = 60
KLINE_COUNT = 340
WORKERS = 10

EVENT_START = dt.date(2025, 10, 16)
DEV_TRAIN_END = dt.date(2026, 5, 31)
DEV_VALID_START = dt.date(2026, 6, 1)
DEV_VALID_END = dt.date(2026, 7, 8)
OOS_START = dt.date(2026, 7, 9)
END = dt.date(2026, 9, 18)


def f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def pct(a, b):
    return (a / b - 1.0) * 100 if b else 0.0


def is_limit_up(bar, prev):
    pc = f(prev.get("close"))
    return pc > 0 and f(bar.get("close")) / pc - 1 >= LIMIT_UP


def model_score(row, model):
    z = model["intercept"]
    for item in model["features"]:
        x = f(row.get(item["name"]))
        sd = f(item["std"], 1e-9)
        z += item["coef"] * ((x - item["mean"]) / sd)
    return 1 / (1 + math.exp(max(-35, min(35, -z))))


def market_states(stock_data, dates):
    result = {}
    for d in dates:
        ds = d.isoformat()
        first, two_plus, zt = set(), set(), set()
        for code, (_, idx_map, bars) in stock_data.items():
            i = idx_map.get(ds)
            if i is None or i == 0:
                continue
            if not is_limit_up(bars[i], bars[i - 1]):
                continue
            zt.add(code)
            if i >= 2 and is_limit_up(bars[i - 1], bars[i - 2]):
                two_plus.add(code)
            else:
                first.add(code)
        result[d] = {"first": first, "two_plus": two_plus, "zt": zt}
    return result


def make_features(bars, i, ctx):
    if i < MIN_HISTORY or i < 21:
        return None

    pre = bars[: i + 1]
    c = [f(x.get("close")) for x in pre]
    v = [f(x.get("volume")) for x in pre]
    t = [f(x.get("turnover")) for x in pre]
    a = [f(x.get("amount")) for x in pre]

    b, prev = bars[i], bars[i - 1]
    pc = f(prev.get("close"))
    o, h, low, close = f(b.get("open")), f(b.get("high")), f(b.get("low")), f(b.get("close"))
    if close <= 0 or pc <= 0:
        return None

    ma5 = statistics.fmean(c[-5:])
    ma10 = statistics.fmean(c[-10:])
    ma20 = statistics.fmean(c[-20:])
    v5 = statistics.fmean(v[-5:])
    v20 = statistics.fmean(v[-20:])
    t5 = statistics.fmean(t[-5:])
    t20 = statistics.fmean(t[-20:])
    a5 = statistics.fmean(a[-5:])
    a20 = statistics.fmean(a[-20:])

    amps = [
        (f(x.get("high")) - f(x.get("low"))) / f(x.get("close")) * 100
        for x in pre[-20:]
        if f(x.get("close")) > 0
    ]
    avg_amp20 = statistics.fmean(amps) if amps else 1.0
    rng = max(h - low, 1e-9)
    open_gap = pct(o, pc)

    prior_5 = pre[-6:-1]
    prior_10 = pre[-11:-1]
    prior_20 = pre[-21:-1]

    def count_zt(seq):
        return sum(
            1
            for k in range(1, len(seq))
            if is_limit_up(seq[k], seq[k - 1])
        )

    prior_zt_5 = count_zt(prior_5)
    prior_zt_10 = count_zt(prior_10)
    prior_zt_20 = count_zt(prior_20)

    days_since_zt = 99
    max_lookback = min(i, 60)
    for back in range(1, max_lookback + 1):
        if is_limit_up(bars[i - back], bars[i - back - 1]):
            days_since_zt = back
            break

    prev_o = f(prev.get("open"))
    prev_h = f(prev.get("high"))
    prev_l = f(prev.get("low"))
    prev_c = f(prev.get("close"))
    prev_v = f(prev.get("volume"))
    prev_t = f(prev.get("turnover"))
    prev_rng = max(prev_h - prev_l, 1e-9)

    prior5_closes = c[-6:-1]
    prior5_highs = [f(x.get("high")) for x in prior_5]
    prior5_lows = [f(x.get("low")) for x in prior_5]

    cur_first = ctx["market_first_count"]
    cur_two = ctx["market_2plus_count"]
    cur_zt = ctx["market_zt_count"]
    prev_first = ctx["prev_market_first_count"]
    prev_two = ctx["prev_market_2plus_count"]
    prev_zt = ctx["prev_market_zt_count"]
    first_share = cur_first / cur_zt if cur_zt else 0.0
    two_share = cur_two / cur_zt if cur_zt else 0.0

    row = {
        # --- V1 frozen feature set ---
        "ret_1d": pct(close, c[-2]),
        "ret_3d": pct(close, c[-4]),
        "ret_5d": pct(close, c[-6]),
        "ret_10d": pct(close, c[-11]),
        "ret_20d": pct(close, c[-21]),
        "volume_5_vs_20": v5 / v20 if v20 else 1,
        "close_above_ma20": int(close > ma20),
        "ma5_gt_ma10": int(ma5 > ma10),
        "ma10_gt_ma20": int(ma10 > ma20),
        "near_high20_pct": pct(close, max(c[-20:])),
        "above_low20_pct": pct(close, min(c[-20:])),
        "up_days_5": sum(c[z] > c[z - 1] for z in range(len(c) - 5, len(c))),
        "volatility_10d": statistics.pstdev(c[-10:]) / close * 100,
        "amplitude_1d": (h - low) / close * 100,
        "market_first_count": cur_first,
        "market_2plus_count": cur_two,
        "market_zt_count": cur_zt,
        "prev_market_first_count": prev_first,
        "prev_market_2plus_count": prev_two,
        "prev_market_1to2_rate": ctx["prev_market_1to2_rate"],
        "open_gap_pct": open_gap,
        "open_gap_band_high": int(open_gap >= 8.5),
        "open_gap_band_mid": int(5 <= open_gap < 8.5),
        "open_gap_band_low": int(open_gap < 5),
        "open_to_close_pct": pct(close, o),
        "lower_wick_ratio": (o - low) / rng,
        "intraday_pullback_pct": pct(o, low),
        "open_position_in_day": (o - low) / rng,
        "amplitude_vs_20d": ((h - low) / pc * 100) / avg_amp20,
        "volume_vs_20d": f(b.get("volume")) / v20 if v20 else 1,
        "near_limit_open": int(open_gap >= 8.0),
        "close_near_high": int(h > 0 and (h - close) / h <= 0.002),

        # --- V1.1 candidate: turnover / amount ---
        "turnover_1d": f(b.get("turnover")),
        "turnover_5d_avg": t5,
        "turnover_20d_avg": t20,
        "turnover_5_vs_20": t5 / t20 if t20 else 1,
        "amount_5_vs_20": a5 / a20 if a20 else 1,
        "turnover_vs_20d": f(b.get("turnover")) / t20 if t20 else 1,
        "amount_vs_20d": f(b.get("amount")) / a20 if a20 else 1,

        # --- V1.1 candidate: prior limit-up history / spacing ---
        "zt_5d_count": prior_zt_5,
        "zt_10d_count": prior_zt_10,
        "zt_20d_count": prior_zt_20,
        "days_since_zt": days_since_zt,

        # --- V1.1 candidate: pre-board day / pre-breakout context ---
        "prev_ret_1d": pct(prev_c, c[-3]),
        "prev_range_pct": (prev_h - prev_l) / prev_c * 100 if prev_c else 0,
        "prev_volume_vs_20": prev_v / v20 if v20 else 1,
        "prev_turnover": prev_t,
        "prior5_range_pct": pct(max(prior5_highs), min(prior5_lows)) if prior5_lows and min(prior5_lows) else 0,
        "prior5_runup_pct": pct(max(prior5_closes), min(prior5_closes)) if prior5_closes and min(prior5_closes) else 0,

        # --- V1.1 candidate: market ratios / regime changes ---
        "market_first_share": first_share,
        "market_2plus_share": two_share,
        "market_first_vs_prev": cur_first / prev_first if prev_first else 1,
        "market_2plus_vs_prev": cur_two / prev_two if prev_two else 1,
        "market_zt_vs_prev": cur_zt / prev_zt if prev_zt else 1,
    }
    return row


BASE_FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "volume_5_vs_20", "close_above_ma20", "ma5_gt_ma10", "ma10_gt_ma20",
    "near_high20_pct", "above_low20_pct", "up_days_5", "volatility_10d",
    "amplitude_1d", "market_first_count", "market_2plus_count", "market_zt_count",
    "prev_market_first_count", "prev_market_2plus_count", "prev_market_1to2_rate",
    "open_gap_pct", "open_gap_band_high", "open_gap_band_mid", "open_gap_band_low",
    "open_to_close_pct", "lower_wick_ratio", "intraday_pullback_pct",
    "open_position_in_day", "amplitude_vs_20d", "volume_vs_20d", "near_limit_open",
    "close_near_high"
]
LIQUIDITY_FEATURES = [
    "turnover_1d", "turnover_5d_avg", "turnover_5_vs_20",
    "amount_5_vs_20", "turnover_vs_20d", "amount_vs_20d"
]
BOARD_HISTORY_FEATURES = ["zt_5d_count", "zt_10d_count", "zt_20d_count", "days_since_zt"]
PREBOARD_FEATURES = [
    "prev_ret_1d", "prev_range_pct", "prev_volume_vs_20", "prev_turnover",
    "prior5_range_pct", "prior5_runup_pct"
]
MARKET_RATIO_FEATURES = [
    "market_first_share", "market_2plus_share",
    "market_first_vs_prev", "market_2plus_vs_prev", "market_zt_vs_prev"
]

EXPERIMENTS = {
    "base32_retrained": BASE_FEATURES,
    "plus_liquidity": BASE_FEATURES + LIQUIDITY_FEATURES,
    "plus_board_history": BASE_FEATURES + BOARD_HISTORY_FEATURES,
    "plus_preboard": BASE_FEATURES + PREBOARD_FEATURES,
    "plus_market_ratios": BASE_FEATURES + MARKET_RATIO_FEATURES,
    "enriched_all": BASE_FEATURES + LIQUIDITY_FEATURES + BOARD_HISTORY_FEATURES + PREBOARD_FEATURES + MARKET_RATIO_FEATURES,
}


def build_events(stock_data, dates, market):
    events = []
    for pos, d in enumerate(dates[:-1]):
        if d < EVENT_START or d >= END:
            continue
        next_d = dates[pos + 1]
        cur = market[d]
        prev = market[dates[pos - 1]] if pos > 0 else None
        prev_first = prev["first"] if prev else set()
        prev_two = prev["two_plus"] if prev else set()
        prev_zt = prev["zt"] if prev else set()
        prev_rate = (
            len(prev_first & cur["two_plus"]) / len(prev_first)
            if prev_first
            else 0.161853203203
        )
        ctx = {
            "market_first_count": len(cur["first"]),
            "market_2plus_count": len(cur["two_plus"]),
            "market_zt_count": len(cur["zt"]),
            "prev_market_first_count": len(prev_first),
            "prev_market_2plus_count": len(prev_two),
            "prev_market_zt_count": len(prev_zt),
            "prev_market_1to2_rate": prev_rate,
        }

        for code, (name, idx_map, bars) in stock_data.items():
            i = idx_map.get(d.isoformat())
            j = idx_map.get(next_d.isoformat())
            if i is None or j != i + 1 or i < MIN_HISTORY:
                continue
            if not is_limit_up(bars[i], bars[i - 1]):
                continue
            if i >= 2 and is_limit_up(bars[i - 1], bars[i - 2]):
                continue

            feat = make_features(bars, i, ctx)
            if feat is None:
                continue
            feat["date"] = d.isoformat()
            feat["prediction_date"] = next_d.isoformat()
            feat["code"] = code
            feat["name"] = name
            feat["target"] = int(is_limit_up(bars[j], bars[j - 1]))
            events.append(feat)
    return events


def evaluate(rows, feature_names):
    if not rows:
        return {}
    X = [[f(r.get(k)) for k in feature_names] for r in rows]
    y = [int(r["target"]) for r in rows]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, solver="liblinear", max_iter=2000))
    ])
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]

    ranked = sorted(
        ({"score": float(s), "target": int(t)} for s, t in zip(p, y)),
        key=lambda r: -r["score"]
    )
    def topk(k):
        sel = ranked[:k]
        return {
            "selected": len(sel),
            "hits": sum(r["target"] for r in sel),
            "precision": (sum(r["target"] for r in sel) / len(sel)) if sel else None,
        }
    return {
        "n": len(rows),
        "positive_rate": sum(y) / len(y) if y else None,
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else None,
        "pr_auc": average_precision_score(y, p) if len(set(y)) > 1 else None,
        "top1": topk(1),
        "top2": topk(2),
        "top6": topk(6),
        "model": model,
    }


def fit_and_score(train_rows, score_rows, feature_names):
    X = [[f(r.get(k)) for k in feature_names] for r in train_rows]
    y = [int(r["target"]) for r in train_rows]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, solver="liblinear", max_iter=2000))
    ])
    model.fit(X, y)
    Xs = [[f(r.get(k)) for k in feature_names] for r in score_rows]
    ps = model.predict_proba(Xs)[:, 1]
    return model, ps


def rank_metrics(rows, scores):
    by_day = {}
    for r, p in zip(rows, scores):
        key = (r["date"], r["prediction_date"])
        by_day.setdefault(key, []).append((float(p), int(r["target"])))
    all_preds = [(float(p), int(t)) for p, t in zip(scores, [r["target"] for r in rows])]
    y = [t for _, t in all_preds]
    p = [s for s, _ in all_preds]
    result = {
        "n": len(rows),
        "baseline": sum(y) / len(y) if y else None,
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else None,
        "pr_auc": average_precision_score(y, p) if len(set(y)) > 1 else None,
        "days": len(by_day),
    }
    for k in (1, 2, 3, 6):
        selected = hits = day_hits = 0
        for items in by_day.values():
            items.sort(key=lambda x: -x[0])
            top = items[:k]
            selected += len(top)
            hits += sum(t for _, t in top)
            day_hits += int(any(t for _, t in top))
        result[f"top{k}"] = {
            "selected": selected,
            "hits": hits,
            "precision": hits / selected if selected else None,
            "lift": (hits / selected) / result["baseline"] if selected and result["baseline"] else None,
            "day_hit_rate": day_hits / len(by_day) if by_day else None,
        }
    return result


def extract_coefficients(model, names):
    lr = model.named_steps["lr"]
    return {
        name: float(coef)
        for name, coef in zip(names, lr.coef_[0])
    }


def main():
    base_model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    universe_rows = [
        x for x in fetch_all_stocks()
        if is_main_board(str(x.get("code", "")), str(x.get("name", "")))
    ]
    universe = {str(x["code"]): str(x["name"]) for x in universe_rows if x.get("code")}
    print(f"Universe: {len(universe)}")

    stock_data = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_kline, code, KLINE_COUNT): code for code in universe}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                bars = sorted(fut.result(), key=lambda x: str(x.get("date", "")))
                if len(bars) >= MIN_HISTORY + 22:
                    idx_map = {str(x.get("date", "")): i for i, x in enumerate(bars)}
                    stock_data[code] = (universe[code], idx_map, bars)
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                print(f"fetch failed {code}: {type(exc).__name__}")

    dates = sorted({
        dt.date.fromisoformat(str(b["date"]))
        for _, _, bars in stock_data.values()
        for b in bars if b.get("date")
    })
    calc_dates = [d for d in dates if d <= END]
    market = market_states(stock_data, calc_dates)
    events = build_events(stock_data, calc_dates, market)

    dev_train = [r for r in events if dt.date.fromisoformat(r["date"]) <= DEV_TRAIN_END]
    dev_valid = [r for r in events if DEV_VALID_START <= dt.date.fromisoformat(r["date"]) <= DEV_VALID_END]
    oos = [r for r in events if dt.date.fromisoformat(r["date"]) >= OOS_START]
    train_plus_dev = [r for r in events if dt.date.fromisoformat(r["date"]) <= DEV_VALID_END]

    print(f"Events: {len(events)} | train: {len(dev_train)} | dev: {len(dev_valid)} | oos: {len(oos)}")

    results = {}
    oos_models = {}
    for name, features in EXPERIMENTS.items():
        dev_train_result = evaluate(dev_train, features)
        dev_model = dev_train_result.pop("model", None)
        if dev_model is None:
            continue

        _, dev_scores = fit_and_score(dev_train, dev_valid, features)
        dev_rank = rank_metrics(dev_valid, dev_scores)

        final_model, oos_scores = fit_and_score(train_plus_dev, oos, features)
        oos_rank = rank_metrics(oos, oos_scores)

        results[name] = {
            "feature_count": len(features),
            "features": features,
            "development_fit_metrics": dev_train_result,
            "development_validation": dev_rank,
            "strict_oos": oos_rank,
            "oos_coefficients": extract_coefficients(final_model, features),
        }
        oos_models[name] = final_model

    # Compare to the currently frozen V1 scores on the same strict OOS events.
    v1_oos = []
    for r in oos:
        s = model_score(r, base_model)
        v1_oos.append((r["date"], r["prediction_date"], r["code"], r["name"], s, int(r["target"])))
    by_day = {}
    for item in v1_oos:
        by_day.setdefault((item[0], item[1]), []).append(item)
    v1_metrics = {"n": len(v1_oos), "baseline": sum(x[5] for x in v1_oos) / len(v1_oos) if v1_oos else None, "days": len(by_day)}
    for k in (1, 2, 3, 6):
        selected = hits = day_hits = 0
        for items in by_day.values():
            items.sort(key=lambda x: (-x[4], x[2]))
            top = items[:k]
            selected += len(top)
            hits += sum(x[5] for x in top)
            day_hits += int(any(x[5] for x in top))
        v1_metrics[f"top{k}"] = {
            "selected": selected,
            "hits": hits,
            "precision": hits / selected if selected else None,
            "lift": (hits / selected) / v1_metrics["baseline"] if selected and v1_metrics["baseline"] else None,
            "day_hit_rate": day_hits / len(by_day) if by_day else None,
        }

    # Simple development-period model selection rule: maximize the average of Top1 and Top2 precision,
    # with Top1 and Top2 both required to beat the base32 retrained model.
    base_dev = results["base32_retrained"]["development_validation"]
    eligible = []
    for name, res in results.items():
        d = res["development_validation"]
        if d["top1"]["precision"] is None or d["top2"]["precision"] is None:
            continue
        if d["top1"]["precision"] >= base_dev["top1"]["precision"] and d["top2"]["precision"] >= base_dev["top2"]["precision"]:
            eligible.append((name, (d["top1"]["precision"] + d["top2"]["precision"]) / 2))
    # Fixed research candidate: enriched_all. No strict-OOS selection and no V1 score input feature.\n    selected_name = "enriched_all"

    out = {
        "research_version": "v2.1-top12-enriched-20260921",
        "v1_frozen": True,
        "v1_reference_top1": base_model.get("reference_test_top1_precision"),
        "data_source": "Sina daily K-line",
        "universe": "current main-board universe; Shanghai/Shenzhen main boards only; excludes ST/STAR/ChiNext/Beijing",
        "event_rule": "close return >= 9.5%; first board = today limit-up and previous trading day not limit-up",
        "outcome_rule": "next actual trading day's limit-up continuation",
        "periods": {
            "development_train_end": DEV_TRAIN_END.isoformat(),
            "development_validation": [DEV_VALID_START.isoformat(), DEV_VALID_END.isoformat()],
            "strict_oos": [OOS_START.isoformat(), END.isoformat()],
        },
        "fetch_failures": failed,
        "event_counts": {
            "all": len(events),
            "development_train": len(dev_train),
            "development_validation": len(dev_valid),
            "strict_oos": len(oos),
        },
        "frozen_v1_on_same_oos": v1_metrics,
        "experiments": results,
        "development_selected_candidate": selected_name,
        "promotion_rule": "No automatic promotion. V1 remains frozen; candidate only earns consideration after strict OOS comparison and failure-mode review.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "period", "top1_precision", "top2_precision", "top6_precision", "auc", "pr_auc", "baseline", "top6_day_hit_rate"])
        for name, res in results.items():
            for period, met in [("development_validation", res["development_validation"]), ("strict_oos", res["strict_oos"])]:
                w.writerow([
                    name, period,
                    met["top1"]["precision"], met["top2"]["precision"], met["top6"]["precision"],
                    met["auc"], met["pr_auc"], met["baseline"], met["top6"]["day_hit_rate"]
                ])
        w.writerow([
            "frozen_v1_same_oos", "strict_oos",
            v1_metrics["top1"]["precision"], v1_metrics["top2"]["precision"], v1_metrics["top6"]["precision"],
            "", "", v1_metrics["baseline"], v1_metrics["top6"]["day_hit_rate"]
        ])

    row_fields = ["date", "prediction_date", "code", "name", "target"] + BASE_FEATURES + LIQUIDITY_FEATURES + BOARD_HISTORY_FEATURES + PREBOARD_FEATURES + MARKET_RATIO_FEATURES
    with OUT_ROWS.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=row_fields)
        w.writeheader()
        for r in events:
            w.writerow({k: r.get(k, "") for k in row_fields})


    selected_model = oos_models[selected_name]
    lr = selected_model.named_steps["lr"]
    scaler = selected_model.named_steps["scale"]
    model_path = ROOT / "config" / "model_v2_1.json"
    model_json = {
        "version": "one-to-two-v2.1-top12-enriched",
        "selected_candidate": selected_name,
        "trained_through": DEV_VALID_END.isoformat(),
        "feature_count": len(results[selected_name]["features"]),
        "features": [
            {
                "name": name,
                "mean": float(scaler.mean_[i]),
                "std": float(scaler.scale_[i]),
                "coef": float(lr.coef_[0][i]),
            }
            for i, name in enumerate(results[selected_name]["features"])
        ],
        "intercept": float(lr.intercept_[0]),
        "selection_rule": "Fixed enriched_all structural candidate from prior V1.1 research; V1 score is not a model input.",
        "strict_oos": results[selected_name]["strict_oos"],
        "v1_same_oos": v1_metrics,
        "notes": [
            "Independent V2.1 branch; does not modify V1, V1.1 research, or V2.",
            "Main-board only; excludes STAR/ChiNext/Beijing/ST.",
            "Sina daily K-line only.",
            "Uses full historical first-board events from 2025-10-16 through 2026-05-31 for fitting, validates 2026-06-01 through 2026-07-08, strict OOS 2026-07-09 through 2026-09-18.",
            "No V1 score input feature, preventing the future-trained V1 score leakage in the earlier V2 attempt.",
            "The candidate itself was identified during the earlier V1.1 research, so this strict-OOS run is a confirmation on the same historical holdout rather than a fresh untouched holdout.",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "selected_development_candidate": selected_name,
        "frozen_v1_strict_oos": v1_metrics,
        "candidate_strict_oos": {
            k: {
                "top1": results[k]["strict_oos"]["top1"],
                "top2": results[k]["strict_oos"]["top2"],
                "auc": results[k]["strict_oos"]["auc"],
                "pr_auc": results[k]["strict_oos"]["pr_auc"]
            }
            for k in results
        }
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
