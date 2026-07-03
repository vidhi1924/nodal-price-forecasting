import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("ercot_analysis_output.csv", parse_dates=["timestamp"])

# Hour needs to be one-hot / cyclical, not linear, to capture the daily curve shape
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

features = ["hour_sin", "hour_cos", "is_weekend", "rolling_mean_24"]
split = int(len(df)*0.85)
train, test = df.iloc[:split], df.iloc[split:]
test_normal = test[~test["is_spike"]]

model = LinearRegression().fit(train[features], train["price_usd_mwh"])
preds = model.predict(test_normal[features])
mae = mean_absolute_error(test_normal["price_usd_mwh"], preds)
r2 = r2_score(test_normal["price_usd_mwh"], preds)
print(f"Baseline model (cyclical hour features), non-spike hours only:")
print(f"MAE: ${mae:.2f}/MWh   R^2: {r2:.3f}")

fig, ax = plt.subplots(figsize=(13,4))
ax.plot(test_normal["timestamp"], test_normal["price_usd_mwh"], label="Actual", color="black", lw=1)
ax.plot(test_normal["timestamp"], preds, label="Forecast", color="orange", lw=1)
ax.set_title(f"Baseline Price Forecast vs Actual (MAE=${mae:.2f}/MWh, R²={r2:.3f})")
ax.set_ylabel("$/MWh"); ax.legend()
plt.tight_layout(); plt.savefig("final_forecast.png", dpi=130)
print("saved final_forecast.png")
