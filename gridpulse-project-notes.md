# GridPulse — Full Project Walkthrough & Notes

*Personal study notes: what the project does, why it was built, how the code works, and the core concepts behind it.*

---

## 1. Why did I create this project?

I wanted a project that showed real quantitative/analytical thinking — not just dashboards, but working with **time-series data, forecasting, and statistical anomaly detection**, since these are core skills for data analyst/data science roles in finance and energy trading.

I specifically chose **nodal electricity prices** because:
- It's a genuinely interesting real-world problem: prices are driven by physics (electricity can't be stored, so supply and demand must balance every instant), which makes the data behavior meaningful, not arbitrary.
- It's directly relevant to companies working in **energy trading** (e.g., roles working with nodal markets like ERCOT), where forecasting and volatility detection are exactly the kind of work analysts do.
- It let me build something in a short time (a couple of hours) while still producing an honest, defensible result — not a toy example, but not an overengineered one either.

For this first version, I built the project on a **simulated dataset engineered to mirror real ERCOT price behavior** (daily/weekly seasonality + random scarcity-price spikes), and disclosed that clearly. The modeling logic doesn't depend on where the data comes from — real ERCOT data could be swapped in without changing the forecasting or detection code. A natural next step is pulling real ERCOT Settlement Point Price data (publicly available) into the same pipeline.

---

## 2. The core problem, in plain terms

Electricity prices behave in **two different ways**:

1. **Predictable pattern** — prices are cheap overnight and expensive in the evening (a daily rhythm), and cheaper on weekends. This repeats.
2. **Unpredictable spikes** — occasionally, a price jumps way up because of something unexpected (an outage, extreme weather, a supply shock). These don't follow a schedule.

The mistake most people make is trying to build **one model** to predict both. This project deliberately builds **two separate tools** instead — one for each behavior — because they need fundamentally different approaches.

---

## 3. Step-by-step: what the code does (no code shown, just the logic)

### Step 1 — Generate realistic fake price data
Since I couldn't pull real ERCOT data, I wrote logic to generate a **90-day, hour-by-hour price series** that mimics real electricity price behavior:
- A repeating daily curve (low overnight, peak in the evening) — shaped using wave-like math functions to mimic the real "duck curve" pattern seen in real grids.
- A slight discount on weekends (lower demand).
- A small upward drift over time (mimicking seasonal demand growth).
- Random small noise added, so it's not a perfectly smooth, unrealistic curve.
- Random price **spikes** injected into about 2% of hours, with large random dollar amounts added — simulating rare, unpredictable price shocks.

### Step 2 — Build "features" (inputs the model can learn from)
A model can't learn from raw timestamps — it needs numeric clues. I created:
- Hour of day, day of week, weekend flag (calendar information)
- The price from 1 hour ago, and the price from the same hour yesterday (recent history)
- A rolling 24-hour average and rolling 24-hour volatility (recent trend and recent noisiness)

### Step 3 — Split the data into "training" and "testing" sets — by time, not randomly
I trained the model on the earliest 85% of days, and tested it on the most recent 15%. This matters because in time-series problems, a model should only ever learn from the past to predict the future — never the other way around. Random splitting would let the model "cheat" by learning from data that comes after what it's predicting.

### Step 4 — Train a baseline forecasting model
I used a simple linear regression model — the most basic prediction approach — to forecast "normal" prices based on the features above. I intentionally started simple rather than jumping to a complex model, to establish an honest baseline first.

### Step 5 — Measure how well it did
I checked two things: on average, how far off (in dollars) were the predictions, and how much of the price pattern did the model actually explain. Both are explained in detail in Section 5 below.

### Step 6 — Build a separate tool to catch spikes (not predict them — detect them)
Instead of asking the forecasting model to also catch spikes, I built a separate, much simpler detector: for every hour, compare the current price to the recent 24-hour average and volatility. If the current price is abnormally far above that recent baseline, flag it as a spike. This doesn't try to predict spikes in advance — it just recognizes them the moment they happen.

### Step 7 — Discover the key insight
When I checked how well the forecasting model performed *specifically on the spike hours*, it did almost no better than a random guess. This isn't a flaw — it's the actual finding: price spikes are caused by events a calendar can't see (outages, weather), so no amount of "what time is it" information can predict them. This is *why* the project splits the problem into two tools instead of forcing one model to do both jobs.

### Step 8 — Refine the model with a smarter way of representing "hour of day"
I noticed a subtle problem: feeding "hour" into the model as a plain number (0–23) makes hour 23 (11pm) and hour 0 (midnight) look mathematically very far apart, when in reality they're only 1 hour apart — the clock wraps around. I fixed this using a standard trick (representing the hour as a position on a circle using sine/cosine), which improved the model's accuracy.

### Step 9 — Re-test the improved model, only on non-spike hours
Since the goal of the forecasting model was always to predict *normal* prices (not spikes — that's the detector's job), I evaluated it fairly by excluding spike hours from the accuracy check. This gave a clearer, more honest picture of how good the baseline forecaster actually is.

### Step 10 — Visualize everything
I generated charts showing: the full price series with detected spikes marked in red, and a close-up comparing the model's forecast against actual prices on the held-out test period.

---

## 4. The core finding (the most important takeaway)

**Two different problems need two different tools.**
- A forecasting model is good at predicting the *predictable* part of prices (time-of-day patterns).
- A statistical anomaly detector is good at catching the *unpredictable* part (spikes) — not by predicting them in advance, but by recognizing them the instant they deviate from recent normal behavior.

Trying to force one model to do both jobs produces a worse, less honest result than building two focused tools.

---

## 5. Key concepts explained — for my notes

### What is MAE (Mean Absolute Error)?
MAE answers: **"On average, how far off were my predictions, in real units (dollars)?"**

How it works: for every prediction, calculate the difference between the predicted price and the actual price (ignore whether it was too high or too low — just the size of the miss). Then average all those differences.

**Example:** if MAE = $12.66/MWh, it means, on average, the model's price guesses were off by about $12.66 in either direction.

**Why it's useful:** it's easy to understand in plain terms — a dollar amount, exactly like the thing you're predicting. Lower MAE = better predictions.

---

### What is R² (R-squared)?
R² answers a different question: **"How much of the pattern/variation in the data did my model actually explain, compared to just guessing the average every time?"**

- R² = 1 → the model perfectly explains all the variation (perfect predictions)
- R² = 0 → the model is no better than just always guessing the average price
- R² can even go **negative** → the model is doing *worse* than a naive "always guess the average" approach

**Why R² was low in this project even though MAE looked decent:** R² depends on how much the actual data varies in the first place. On "normal" (non-spike) hours, prices don't swing around all that much — there isn't a huge amount of variation to explain. So even a fairly accurate model (low MAE) can end up with a low R², simply because there wasn't much variation there to take credit for. This doesn't mean the model failed — it means **MAE is the more honest, informative metric to lead with for this particular dataset**, since R² can be misleading on low-variation data.

**Rule of thumb for future reference:** MAE tells you "how big are my errors, in real terms." R² tells you "how much better am I than a lazy guess." Both matter, but they can tell different stories — always look at both, and understand *why* they might disagree.

---

### What is a Z-score (used for spike detection)?
A z-score answers: **"How unusual is this value, compared to what's been normal recently?"**

It's calculated as: *(current value − recent average) ÷ recent volatility (standard deviation)*.

- Z-score of 0 → exactly at the recent average, totally normal
- Z-score of 1–2 → somewhat above average, not unusual
- Z-score above 3 → statistically rare — in this project, treated as a "spike"

**Why it works well for spike detection:** it automatically adapts to what "normal" has looked like *recently* (using a rolling 24-hour window), rather than comparing to some fixed threshold. So it correctly identifies abnormal hours even as the overall price level drifts over time.

---

### Why time-based train/test split instead of random split?
In most machine learning problems, data is split randomly into training and testing sets. But for **time-series data** (anything ordered by time, like prices), this is wrong — a random split could let the model accidentally learn from future data to predict the past, which is impossible in real life and produces misleadingly good results. The correct approach is to always train on earlier data and test on later data, mimicking how the model would actually be used in the real world: predicting the future from the past.

---

### Why cyclical (sine/cosine) encoding for "hour of day"?
Representing hour of day as a plain number (0 to 23) creates a false discontinuity: hour 23 and hour 0 are only 1 hour apart in real life, but as plain numbers, they look 23 units apart to a model. Converting the hour into a position on a circle using sine and cosine fixes this — it's a standard technique for any cyclical value (hour of day, day of week, month of year, compass direction, etc.).

---

## 6. Honest limitations (things I should be ready to explain)

- **The dataset is simulated**, not real ERCOT data — built to mirror documented ERCOT price behavior, but not pulled from a live feed. This is disclosed openly in the project's README. Real ERCOT price data is publicly available and could be pulled in as a next step without changing the modeling code.
- **Linear regression is a simple/baseline model** — it was chosen deliberately to establish an honest starting point, not because it's the most sophisticated option. A logical next step would be trying more advanced time-series models (e.g., ARIMA, gradient boosting) to see if they meaningfully improve on this baseline.
- **R² near zero on spikes is a genuine finding, not a bug** — it demonstrates that spikes are driven by information the model doesn't have access to (outages, weather shocks), not a modeling mistake.

---

## 7. 30-second summary (for verbal explanation)

"I simulated realistic hourly electricity prices with a daily/weekly pattern plus random spikes as a first version, with real ERCOT data as a natural next step. I built a simple linear regression to forecast the predictable part, using time-of-day (encoded cyclically) and recent rolling averages as inputs, trained on the earliest 85% of days and tested on the most recent 15% — never letting the model see the future. Separately, I used a rolling z-score to flag abnormal spikes as they happen, rather than trying to predict them, because testing proved spikes aren't predictable from calendar data alone — R² near zero. That's the core finding: two different problems need two different tools."
