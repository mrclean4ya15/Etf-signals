#!/usr/bin/env python3
"""
ETF Swing-Trade Signal Engine
------------------------------
Pulls recent price history for a watchlist of leveraged ETFs, computes
RSI / MACD / Bollinger Bands / volume signals, and pushes a Pushover
notification when a buy or sell condition triggers.

This is a technical-indicator tool, not investment advice. Indicators
lag price and can whipsaw, especially on leveraged/volatile ETFs.
Treat every alert as a prompt to look at the chart yourself, not an
instruction to trade.
"""

import os
import json
import sys
from datetime import datetime, timezone
import urllib.request
import urllib.parse

FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY")
PUSHOVER_TOKEN = os.environ.get("PUSHOVER_APP_TOKEN")
PUSHOVER_USER = os.environ.get("PUSHOVER_USER_KEY")

WATCHLIST = ["TQQQ", "SQQQ", "SPXL", "SPXU"]
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BB_PERIOD = 20
BB_STDDEV = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def fetch_candles(symbol, resolution="15", lookback_days=15):
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - lookback_days * 24 * 60 * 60
    params = urllib.parse.urlencode({
        "symbol": symbol,
        "resolution": resolution,
        "from": start,
        "to": now,
        "token": FINNHUB_KEY,
    })
    url = f"https://finnhub.io/api/v1/stock/candle?{params}"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode())
    if data.get("s") != "ok":
        raise RuntimeError(f"No candle data for {symbol}: {data}")
    return data


def sma(values, period):
    return [
        sum(values[i - period + 1:i + 1]) / period if i >= period - 1 else None
        for i in range(len(values))
    ]


def ema_series(values, period):
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(values, period=RSI_PERIOD):
    gains, losses = [0], [0]
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period
    out = [None] * period
    for i in range(period, len(values)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        out.append(100 - (100 / (1 + rs)))
    return out


def macd(values):
    ema_fast = ema_series(values, MACD_FAST)
    ema_slow = ema_series(values, MACD_SLOW)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema_series(macd_line, MACD_SIGNAL)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist


def bollinger(values, period=BB_PERIOD, stddev=BB_STDDEV):
    mid = sma(values, period)
    upper, lower = [], []
    for i in range(len(values)):
        if mid[i] is None:
            upper.append(None)
            lower.append(None)
            continue
        window = values[i - period + 1:i + 1]
        mean = mid[i]
        var = sum((x - mean) ** 2 for x in window) / period
        sd = var ** 0.5
        upper.append(mean + stddev * sd)
        lower.append(mean - stddev * sd)
    return upper, mid, lower


def evaluate(symbol, candles):
    closes = candles["c"]
    vols = candles["v"]
    if len(closes) < max(BB_PERIOD, MACD_SLOW + MACD_SIGNAL) + 5:
        return None

    r = rsi(closes)
    m_line, m_signal, m_hist = macd(closes)
    bb_up, bb_mid, bb_low = bollinger(closes)
    avg_vol = sum(vols[-20:]) / 20

    price = closes[-1]
    r_now, r_prev = r[-1], r[-2]
    hist_now, hist_prev = m_hist[-1], m_hist[-2]
    vol_now = vols[-1]

    score = 0
    reasons = []

    if r_now is not None and r_now < RSI_OVERSOLD:
        score += 1
        reasons.append(f"RSI {r_now:.0f} (oversold)")
    if r_now is not None and r_now > RSI_OVERBOUGHT:
        score -= 1
        reasons.append(f"RSI {r_now:.0f} (overbought)")

    if hist_prev is not None and hist_now is not None:
        if hist_prev < 0 <= hist_now:
            score += 1
            reasons.append("MACD bullish crossover")
        if hist_prev > 0 >= hist_now:
            score -= 1
            reasons.append("MACD bearish crossover")

    if bb_low[-1] is not None and price <= bb_low[-1]:
        score += 1
        reasons.append("price at/below lower Bollinger Band")
    if bb_up[-1] is not None and price >= bb_up[-1]:
        score -= 1
        reasons.append("price at/above upper Bollinger Band")

    if vol_now > avg_vol * 1.5:
        reasons.append(f"volume {vol_now/avg_vol:.1f}x avg")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "signal": signal,
        "score": score,
        "reasons": reasons,
        "rsi": round(r_now, 1) if r_now is not None else None,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_pushover(title, message, priority=0):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        print("Pushover not configured, skipping notification:", title)
        return
    data = urllib.parse.urlencode({
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "title": title,
        "message": message,
        "priority": priority,
    }).encode()
    req = urllib.request.Request("https://api.pushover.net/1/messages.json", data=data)
    with urllib.request.urlopen(req, timeout=15) as r:
        r.read()


def main():
    if not FINNHUB_KEY:
        print("FINNHUB_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    state = load_state()
    results = []

    for symbol in WATCHLIST:
        try:
            candles = fetch_candles(symbol)
            result = evaluate(symbol, candles)
        except Exception as e:
            print(f"Error processing {symbol}: {e}", file=sys.stderr)
            continue
        if result is None:
            continue
        results.append(result)

        prev_signal = state.get(symbol, {}).get("signal", "HOLD")
        if result["signal"] != "HOLD" and result["signal"] != prev_signal:
            why = "; ".join(result["reasons"]) if result["reasons"] else ""
            send_pushover(
                title=f"{symbol}: {result['signal']} signal",
                message=f"${result['price']}  —  {why}",
                priority=0,
            )
            print(f"ALERT sent: {symbol} {result['signal']}")

        state[symbol] = result

    save_state(state)

    with open(os.path.join(os.path.dirname(__file__), "latest.json"), "w") as f:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(), "signals": results}, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
