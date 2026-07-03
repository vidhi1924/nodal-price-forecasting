import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

# ---------------------------------------------------------------
# 1. GENERATE REALISTIC NODAL ELECTRICITY PRICE SERIES
#    Modeled on documented ERCOT real-time SPP behavior:
#    - Strong daily seasonality (low overnight, peak in evening)
#    - Weekly seasonality (lower on weekends)
#    - Summer-like base volatility with occasional extreme price spikes
#    (scarcity pricing events), consistent with ERCOT's real-time market
# ---------------------------------------------------------------
n_days = 90
periods_per_day = 24  # hourly resolution
n = n_days * periods_per_day
start = pd.Timestamp("2026-04-01")
timestamps = pd.date_range(start, periods=n, freq="h")

hour = timestamps.hour.values
dow = timestamps.dayofweek.values
day_idx = np.arange(n) // periods_per_day

# Base daily load-shape curve (duck-curve-like: morning ramp, evening peak)
daily_shape = 25 + 15*np.sin((hour-6)/24*2*np.pi) + 10*np.exp(-((hour-19)**2)/8)
weekend_discount = np.where(np.isin(dow, [5,6]), -6, 0)
trend = 0.05 * day_idx  # slow upward drift (seasonal demand growth)
noise = np.random.normal(0, 4, n)

base_price = daily_shape + weekend_discount + trend + noise
base_price = np.clip(base_price, 5, None)

# Inject scarcity-pricing spikes (~2% of hours), mimicking real ERCOT spike events
spike_mask = np.random.rand(n) < 0.02
spike_magnitude = np.random.uniform(150, 900, n)
price = base_price.copy()
price[spike_mask] += spike_magnitude[spike_mask]

df = pd.DataFrame({"timestamp": timestamps, "price_usd_mwh": price})
df.to_csv("simulated_ercot_prices.csv", index=False)
print(f"Generated {len(df)} hourly price points across {n_days} days")
print(df["price_usd_mwh"].describe())

# ---------------------------------------------------------------
# 2. FEATURE ENGINEERING for forecasting (lag + calendar features)
# ---------------------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["dow"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
df["lag_1"] = df["price_usd_mwh"].shift(1)
df["lag_24"] = df["price_usd_mwh"].shift(24)   # same hour yesterday
df["rolling_mean_24"] = df["price_usd_mwh"].rolling(24).mean()
df["rolling_std_24"] = df["price_usd_mwh"].rolling(24).std()
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------
# 3. TRAIN/TEST SPLIT (time-based, no shuffling — realistic for TS)
# ---------------------------------------------------------------
split = int(len(df) * 0.85)
train, test = df.iloc[:split], df.iloc[split:]

features = ["hour", "dow", "is_weekend", "lag_1", "lag_24", "rolling_mean_24"]
X_train, y_train = train[features], train["price_usd_mwh"]
X_test, y_test = test[features], test["price_usd_mwh"]

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"\nModel: Linear Regression w/ lag + calendar features")
print(f"Test MAE: ${mae:.2f}/MWh")
print(f"Test R^2: {r2:.3f}")

# ---------------------------------------------------------------
# 4. VOLATILITY / PRICE-SPIKE DETECTION (rolling z-score method)
# ---------------------------------------------------------------
df["z_score"] = (df["price_usd_mwh"] - df["rolling_mean_24"]) / df["rolling_std_24"]
df["is_spike"] = df["z_score"] > 3
spike_rate = df["is_spike"].mean() * 100
n_spikes = df["is_spike"].sum()
print(f"\nVolatility detection: {n_spikes} price spikes flagged (z>3) out of {len(df)} hours ({spike_rate:.1f}%)")

# ---------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(13, 8))

axes[0].plot(df["timestamp"], df["price_usd_mwh"], color="steelblue", linewidth=0.7, label="Actual Price")
axes[0].scatter(df.loc[df["is_spike"], "timestamp"], df.loc[df["is_spike"], "price_usd_mwh"],
                 color="red", s=25, zorder=5, label="Detected Spike (z>3)")
axes[0].set_title("Simulated Nodal Electricity Price (Hourly) with Detected Volatility Spikes")
axes[0].set_ylabel("Price ($/MWh)")
axes[0].legend()

axes[1].plot(test["timestamp"], y_test.values, label="Actual", color="black", linewidth=1)
axes[1].plot(test["timestamp"], preds, label="Forecast (Linear Regression)", color="orange", linewidth=1)
axes[1].set_title(f"Test-Period Forecast vs Actual (MAE=${mae:.2f}/MWh, R²={r2:.3f})")
axes[1].set_ylabel("Price ($/MWh)")
axes[1].legend()

plt.tight_layout()
plt.savefig("price_forecast_analysis.png", dpi=130)
print("\nSaved plot: price_forecast_analysis.png")

df.to_csv("ercot_analysis_output.csv", index=False)

# ---------------------------------------------------------------
# 6. REFINEMENT: baseline-price model evaluated on non-spike hours
#    (standard practice: forecast the base curve; handle spikes via
#    separate anomaly detection rather than forcing regression to fit them)
# ---------------------------------------------------------------
test_normal = test[~df.loc[test.index, "is_spike"]]
X_test_n, y_test_n = test_normal[features], test_normal["price_usd_mwh"]
preds_n = model.predict(X_test_n)
mae_n = mean_absolute_error(y_test_n, preds_n)
r2_n = r2_score(y_test_n, preds_n)
print(f"\nBaseline-price model (excluding spike hours):")
print(f"Test MAE: ${mae_n:.2f}/MWh")
print(f"Test R^2: {r2_n:.3f}")
