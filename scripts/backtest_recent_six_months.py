#!/usr/bin/env python3
from __future__ import annotations
import csv, datetime, json, math, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from market_data_sina import fetch_all_stocks, fetch_kline, is_main_board

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "config" / "model_v1.json"
OUT_JSON = ROOT / "reports" / "backtest_recent_six_months.json"
OUT_CSV = ROOT / "reports" / "backtest_recent_six_months.csv"

LIMIT_UP = 0.095
MIN_HISTORY = 60
KLINE_COUNT = 220
WORKERS = 10
TOPKS = (1, 2, 3, 4, 5, 6)
FULL_START = datetime.date(2026, 3, 20)
OOS_START = datetime.date(2026, 7, 9)
END = datetime.date(2026, 9, 18)


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


def make_features(bars, i, ctx):
    if i < MIN_HISTORY or i < 21:
        return None
    pre = bars[:i + 1]
    c = [f(x.get("close")) for x in pre]
    v = [f(x.get("volume")) for x in pre]
    b, prev = bars[i], bars[i - 1]
    pc = f(prev.get("close"))
    o, h, l, close = f(b.get("open")), f(b.get("high")), f(b.get("low")), f(b.get("close"))
    if close <= 0 or pc <= 0:
        return None

    ma5 = statistics.fmean(c[-5:])
    ma10 = statistics.fmean(c[-10:])
    ma20 = statistics.fmean(c[-20:])
    v5 = statistics.fmean(v[-5:])
    v20 = statistics.fmean(v[-20:])
    amps = [
        (f(x.get("high")) - f(x.get("low"))) / f(x.get("close")) * 100
        for x in pre[-20:] if f(x.get("close")) > 0
    ]
    avg_amp20 = statistics.fmean(amps) if amps else 1.0
    rng = max(h - l, 1e-9)
    open_gap = pct(o, pc)

    row = {
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
        "amplitude_vs_20d": ((h - l) / pc * 100) / avg_amp20,
        "volume_vs_20d": f(b.get("volume")) / v20 if v20 else 1,
        "near_limit_open": int(open_gap >= 8.0),
        "close_near_high": int(h > 0 and (h - close) / h <= 0.002),
    }
    return row


def prepare_bars(raw_bars):
    bars = sorted(raw_bars, key=lambda x: str(x.get("date", "")))
    index = {str(x.get("date", "")): i for i, x in enumerate(bars)}
    return bars, index


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


def run_window(start_date, dates, stock_data, market, model):
    window_dates = [d for d in dates if start_date <= d <= END]
    daily = []
    for pos, d in enumerate(window_dates[:-1]):
        next_d = window_dates[pos + 1]
        cur = market[d]
        prev = market[window_dates[pos - 1]] if pos > 0 else None
        prev_first = prev["first"] if prev else set()
        prev_two = prev["two_plus"] if prev else set()
        prev_rate = len(prev_first & cur["two_plus"]) / len(prev_first) if prev_first else model.get("bootstrap_prev_market_1to2_rate", 0.16185320352130533)

        # Do not use the next day's market result for scoring; only individual stock outcome is evaluated.
        candidates = []
        for code, (name, idx_map, bars) in stock_data.items():
            i = idx_map.get(d.isoformat())
            j = idx_map.get(next_d.isoformat())
            if i is None or j != i + 1 or i < MIN_HISTORY:
                continue
            if not is_limit_up(bars[i], bars[i - 1]):
                continue
            if i >= 2 and is_limit_up(bars[i - 1], bars[i - 2]):
                continue

            ctx = {
                "market_first_count": len(cur["first"]),
                "market_2plus_count": len(cur["two_plus"]),
                "market_zt_count": len(cur["zt"]),
                "prev_market_first_count": len(prev_first),
                "prev_market_2plus_count": len(prev_two),
                "prev_market_1to2_rate": prev_rate,
            }
            feat = make_features(bars, i, ctx)
            if feat is None:
                continue
            candidates.append({
                "date": ds if (ds := d.isoformat()) else "",
                "prediction_date": next_d.isoformat(),
                "code": code,
                "name": name,
                "score": model_score(feat, model),
                "is_2board": int(is_limit_up(bars[j], bars[j - 1])),
            })

        candidates.sort(key=lambda x: (-x["score"], x["code"]))
        daily.append({
            "date": d.isoformat(),
            "prediction_date": next_d.isoformat(),
            "first_count": len(candidates),
            "first_success": sum(x["is_2board"] for x in candidates),
            "top": candidates,
        })
    return daily


def aggregate(daily):
    total_first = sum(x["first_count"] for x in daily)
    total_success = sum(x["first_success"] for x in daily)
    baseline = total_success / total_first if total_first else None
    out = {
        "days": len(daily),
        "first_board_events": total_first,
        "successful_first_board_events": total_success,
        "baseline_one_to_two_rate": baseline,
    }
    for k in TOPKS:
        selected = sum(len(x["top"][:k]) for x in daily)
        hits = sum(sum(y["is_2board"] for y in x["top"][:k]) for x in daily)
        day_hit_days = sum(any(y["is_2board"] for y in x["top"][:k]) for x in daily)
        precision = hits / selected if selected else None
        out[f"top{k}"] = {
            "selected": selected,
            "hits": hits,
            "precision": precision,
            "lift_vs_all_first": precision / baseline if precision is not None and baseline else None,
            "day_hit_days": day_hit_days,
            "day_hit_rate": day_hit_days / len(daily) if daily else None,
        }
    return out


def main():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    universe_rows = [
        x for x in fetch_all_stocks()
        if is_main_board(str(x.get("code", "")), str(x.get("name", "")))
    ]
    universe = {str(x["code"]): str(x["name"]) for x in universe_rows if x.get("code")}
    print(f"Current main-board universe: {len(universe)}")

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
            except Exception:
                failed += 1

    # Build a common trading-date axis from the fetched market data.
    dates = sorted({
        datetime.date.fromisoformat(str(b["date"]))
        for _, _, bars in stock_data.values()
        for b in bars if b.get("date")
    })
    # Keep enough pre-history for the 60-bar feature lookback, while scoring only the requested windows.
    calc_dates = [d for d in dates if FULL_START - datetime.timedelta(days=140) <= d <= END]
    if not calc_dates:
        raise RuntimeError("No historical trading dates available")

    print(f"Historical dates available: {calc_dates[0]} -> {calc_dates[-1]} ({len(calc_dates)} trading dates)")
    market = market_states(stock_data, calc_dates)

    full_daily = run_window(FULL_START, calc_dates, stock_data, market, model)
    oos_daily = run_window(OOS_START, calc_dates, stock_data, market, model)

    result = {
        "model_version": model.get("version"),
        "model_trained_through": model.get("trained_through"),
        "data_source": "Sina daily K-line",
        "universe": "current main-board universe only; exclude ST/STAR/ChiNext/Beijing",
        "event_rule": "close return >= 9.5%; first board = today's limit-up and previous trading day not limit-up",
        "outcome_rule": "next actual trading day's limit-up continuation = second board",
        "note": "The recent-six-month window overlaps the model's training period. Use strict_oos for a clean post-training estimate. Historical universe follows the current live-scan universe, so delisted/name-status changes can introduce survivorship bias.",
        "failed_kline_fetches": failed,
        "windows": {
            "recent_6_months": aggregate(full_daily),
            "strict_oos": aggregate(oos_daily),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "date", "prediction_date", "rank", "code", "name", "score", "is_2board"])
        for label, daily in [("recent_6_months", full_daily), ("strict_oos", oos_daily)]:
            for day in daily:
                for rank, row in enumerate(day["top"][:10], 1):
                    w.writerow([label, row["date"], row["prediction_date"], rank, row["code"], row["name"], row["score"], row["is_2board"]])

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
