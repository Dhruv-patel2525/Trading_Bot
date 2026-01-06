# Trading Bot Plan v1 (Research Prototype)

> **Scope:** v1 is a **research + paper-trading prototype** for **Bitget USDT-M futures** on **15-minute candles** using **market orders** and **model-gated entries** (long/short/no-trade).  
> **Goal:** a pipeline you can trust: data → features → labels → model → backtest (with costs) → paper trade.

⚠️ Not financial advice. With leverage, bad risk controls = fast liquidation.

---

## Locked decisions (v1)

- Exchange/market: **Bitget USDT-M perpetual futures**
- Leverage: **10x** (use **isolated margin**)
- Timeframe: **15m**
- Style: **intraday long/short**
- Max holding time: **6 hours (24 candles)**
- Orders: **market orders**
- Entry timing: decide at candle close, enter next candle open (avoid lookahead)
- Exit rules: **TP / SL only**, plus time-out at 6 hours
- One position at a time
- “Setup detected”: **model gate** (trade only above confidence/edge threshold)
- Position sizing: **fixed risk per trade** (recommended default: **1% equity risk**)
- Daily stop: **3R** (stop trading after 3 full-risk losses)

---

## End-to-end workflow

```mermaid
flowchart TD
  A[Trading spec - Bitget USDT-M, 10x, market, intraday] --> B[Costs model - taker fee + slippage bps]
  B --> C[Data sourcing - Binance long history + Bitget recent window]
  C --> D[Clean/align OHLCV - 15m, gaps, timestamps]
  D --> E[Features - causal returns, ATR, vol, volume, trend]
  E --> F[Triple-barrier labels - ATR SL/TP + 6h timeout + costs]
  F --> G[Walk-forward splits - purge/embargo]
  G --> H[Train baseline - XGBoost/LightGBM]
  H --> I[Train sequence model - CNN/LSTM then Transformer]
  I --> J[Model gate - trade only if prob>threshold]
  J --> K[Backtest with intrabar SL/TP - fees+slippage+daily stop]
  K --> L{Robust net PnL?}
  L -- No --> F
  L -- Yes --> M[Paper trade on Bitget - same pipeline]
  M --> N[Small live deploy - 1% risk, 3R daily stop, kill switch]
```
