# Nodal Electricity Price Forecasting & Volatility Detection
📝 Read the full writeup: [Why I Stopped Trying to Predict Electricity Price Spikes] https://medium.com/@shahvidhijayesh/why-i-stopped-trying-to-predict-electricity-price-spikes-and-built-something-better-instead-f2e95f102f91?sharedUserId=shahvidhijayesh

Python pipeline that forecasts baseline nodal electricity prices and separately
flags volatility/price-spike events — modeled on real-world nodal market behavior
(e.g. ERCOT's real-time Settlement Point Price patterns).

## Why two models instead of one

Electricity spot prices have two very different regimes:
- **Baseline price** — driven by predictable daily/weekly demand seasonality
- **Price spikes** — driven by unexpected supply/demand shocks (outages, extreme
  weather, scarcity conditions), which are largely *not* predictable from calendar
  features alone

Rather than forcing a single model to fit both, this project forecasts the baseline
curve with regression and handles spikes with a separate real-time anomaly detector —
the same approach used in practice for nodal price risk monitoring.

## Data

Simulated hourly nodal price series (90 days, 2,160 points), built to reflect
documented ERCOT real-time price behavior: duck-curve daily shape, weekend demand
dip, slow seasonal drift, and stochastic scarcity-pricing spikes.

> This is a simulated dataset built for portfolio purposes — ERCOT's raw price data
> requires MIS portal access. The generation logic (`generate_and_analyze.py`) can be
> swapped for a real ERCOT CSV pull with no changes to the modeling code.

## Method

| Step | Approach |
|---|---|
| Feature engineering | Cyclical (sin/cos) hour encoding, weekend flag, 24h rolling mean/std |
| Baseline forecasting | Linear Regression, time-based 85/15 train/test split |
| Volatility detection | Rolling 24h z-score; hours >3 std devs above rolling mean flagged as spikes |

## Results

- Baseline model: **$12.66/MWh MAE** on non-spike hours
- **34 spikes** flagged (1.6% of hours) via z-score anomaly detection
- Calendar/lag features alone have near-zero explanatory power for spikes (R² ≈ 0) —
  a realistic finding, and the reason spikes are handled by anomaly detection rather
  than forced into the regression target

## Run it

```bash
pip install pandas numpy matplotlib scikit-learn
python generate_and_analyze.py   # generates data, fits baseline model, plots
python final_model.py            # refined model with cyclical hour features
```

## Files

- `generate_and_analyze.py` — data simulation, feature engineering, baseline model, spike detection
- `final_model.py` — refined baseline model with cyclical hour encoding
- `price_forecast_analysis.png`, `final_forecast.png` — output visualizations
