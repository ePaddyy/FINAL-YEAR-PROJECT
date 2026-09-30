# Project Explainer Guide
### Forecasting Fuel Prices in Ghana: ARIMA, ARIMAX, and VAR
*A walkthrough of what was done, why, and what every number means — written so you can explain it confidently to your supervisor.*

---

## 0. The big picture

Your study asks: **can we forecast Ghana's diesel, petrol, and LPG prices, and does adding external information (oil prices, exchange rate) help?**

To answer this, you built a pipeline of 7 notebooks that go in a strict logical order — each one depends on the one before it:

1. **01 — Data cleaning & EDA**: get the raw numbers, check they're trustworthy, look at them visually.
2. **02 — Stationarity testing**: figure out whether the data behaves in a way that classical forecasting models are allowed to assume.
3. **03 — Model identification**: get an initial visual guess at what kind of model structure fits.
4. **04 — ARIMA**: fit a forecasting model using *only* each fuel's own history.
5. **05 — ARIMAX**: fit the same idea, but let the model also see Brent crude oil prices and the exchange rate.
6. **06 — Rolling-origin evaluation**: stress-test all models repeatedly across many time windows, not just once.
7. **07 — VAR**: treat all six variables (three fuels + Brent + exchange rate + CPI) as one interconnected system instead of separate models.

**Chapter 4** is where all of this gets written up in narrative form for the thesis — every number in Chapter 4 traces back to one of these seven notebooks.

---

## 1. Notebook 01 — Data Cleaning & EDA

### What the code does

```python
df = pd.read_csv("full_data.csv")
df['date'] = pd.to_datetime(df['month'], format='%YM%m')
df.set_index('date', inplace=True)
df.index.freq = 'MS'
```

- Your raw data has a `month` column formatted like `"2015M07"` — a custom text format, not a real date.
- `pd.to_datetime(..., format='%YM%m')` tells pandas exactly how to read it: 4-digit year, literal letter "M", 2-digit month.
- `set_index('date')` makes the date the row label instead of a plain column — this is required for every time-series function used later (ADF, ARIMA, plotting).
- `df.index.freq = 'MS'` explicitly tells pandas "this is monthly data, Month-Start." Without this, some statistical functions can't tell how far apart your observations are.

### The custom plotting functions

You wrote five reusable functions instead of writing plotting code six times over:
- `heatmap()` — draws the correlation matrix as a colour grid.
- `hist()` — draws a histogram for a column.
- `line()` — draws a grid of line plots, one per numeric column, arranged automatically into rows/columns.
- `scatter()` — draws one scatter plot between two chosen columns.
- `scatter_grid()` — draws *every possible pair* of columns as small scatter plots in one big grid (uses `itertools.combinations` to generate all pairs without repeats).

**Why this matters**: reusable functions mean you write the plotting logic once and call it six times with different inputs, instead of copy-pasting plotting code and risking small inconsistencies between plots.

### What the dataset actually contains

- **132 monthly rows**, July 2015 to June 2026.
- **6 numeric columns**: diesel price, petrol price, LPG price, Brent crude price, USD/GHS exchange rate, CPI.
- **Zero missing values** — you checked this explicitly with `.isnull().sum()`.

### Figure 4.1 — the 6-panel trend plot

This is `line(df, num_cols)` — one line chart per variable, sharing the same visual layout.

**What it shows**: all six variables rise over time, but the shape of the rise is very different:
- Diesel, petrol, and the exchange rate show a **sharp spike around late 2022**, shoot up dramatically, then partially fall back down.
- LPG rises more like a staircase — flat for years, then a steep climb from about 2022, then flattens again.
- Brent crude has its own independent up-and-down pattern (a COVID crash around 2020, a spike around 2022) that doesn't match the domestic fuel prices' shape as closely.
- CPI just climbs steadily and never comes back down — this is normal for a price index, since inflation rarely reverses.

**Why it matters**: this is your first visual clue that something structurally unusual happened around 2022 — this becomes the central thread of your whole analysis (formally confirmed later by the Zivot-Andrews test).

### Figure 4.2 — the histogram grid

This shows the *distribution* of each variable (how often each value range occurs), rather than its path over time.

**What "bimodal" means and why it matters**: most of your variables show **two separate humps** rather than one smooth bell shape — a cluster of low values and a separate cluster of high values, with a gap in between. This is the *distributional fingerprint* of the same 2022 event you saw in Figure 4.1: before 2022, prices sat in a low range; after 2022, they permanently jumped to a higher range. A variable that changed smoothly over time would show one smooth hump, not two separate ones.

### Figure 4.3 — the correlation heatmap

Each cell shows the **Pearson correlation coefficient** between two variables — a number between −1 and +1 that measures how strongly two variables move together in a straight-line sense. +1 means perfect together-movement, 0 means no relationship, −1 means perfect opposite movement.

**Key numbers**:
- Diesel–Petrol: **0.99** (almost perfectly together — makes sense, both are refined fuel products sold in the same market)
- Diesel–LPG: **0.96**
- Diesel–Brent: **0.67** (much weaker — Brent is a global commodity price, not a domestic pump price)

**The important caveat** (this is something worth saying explicitly to your supervisor): **high correlation between trending variables does not prove a real relationship.** Two completely unrelated variables that are both generally rising over time will show high correlation just because they're both going up — this is called **spurious correlation** (a classic result by Granger & Newbold, 1974). This is exactly why the next notebook (stationarity testing) matters so much: you can't trust these correlation numbers at face value until you've checked whether the underlying trend is being properly accounted for.

---

## 2. Notebook 02 — Stationarity Testing

### The core concept: what is "stationarity" and why does it matter?

A time series is **stationary** if its statistical properties (average level, how much it varies) stay roughly constant over time. Most classical forecasting models — including ARIMA — are built on the mathematical assumption that the series is stationary. If you feed a non-stationary series (one with a strong trend) into these models without fixing that first, the model's assumptions are violated and its output becomes unreliable.

Your fuel price series are clearly **not** stationary — Figure 4.1 shows them trending strongly upward. So before any model can be fit, you need to either transform the data into a stationary form, or use a testing framework that accounts for the trend.

### The three tests, explained

**ADF (Augmented Dickey-Fuller) test**
- **Null hypothesis (H₀)**: the series has a "unit root" — technical language for "the series is non-stationary, and its wandering-away behaviour never settles down."
- **How to read the p-value**: if p < 0.05, you *reject* the null — meaning you have evidence the series **is** stationary. If p ≥ 0.05, you fail to reject — meaning you don't have evidence it's stationary (in practice, this is usually read as "it's probably non-stationary").

**KPSS test**
- **Null hypothesis (H₀)**: the series **is** stationary. This is the *opposite* null hypothesis from ADF — deliberately so.
- **How to read the p-value**: if p < 0.05, you reject the null — meaning the series is **non**-stationary. If p ≥ 0.05, you don't reject — consistent with stationarity.
- **Why use both**: ADF and KPSS can occasionally disagree because they're testing from opposite starting assumptions. Running both and checking if they agree gives you a much more trustworthy conclusion than either alone.

**Zivot-Andrews (ZA) test**
- **Null hypothesis (H₀)**: the series has a unit root — *but* the test explicitly allows for **one structural break** (a single moment where the series' behaviour permanently shifts) somewhere in the sample, and finds the most likely break point automatically.
- **Why this test specifically**: ordinary ADF/KPSS can be fooled by a structural break — a single sharp permanent shift can look statistically identical to a smooth trend, causing these tests to wrongly conclude "non-stationary" when really the series is stable except for one clean jump. ZA is designed to tell these two situations apart.

### What your results actually showed (at level, i.e. raw data)

All six variables: **non-stationary** by both ADF and KPSS. Not surprising, given the visible trends in Figure 4.1.

But the Zivot-Andrews test told a more interesting story:

| Series | ZA conclusion | Estimated break month |
|---|---|---|
| Diesel | Stationary once a break is allowed | January 2022 |
| LPG | Stationary once a break is allowed | March 2022 |
| Petrol | Stationary once a break is allowed | February 2022 |
| Exchange rate | Stationary once a break is allowed | March 2022 |
| Brent crude | Still non-stationary | (2021-01, but not statistically confirmed) |
| CPI | Still non-stationary | (2022-02, but not statistically confirmed) |

**What this means in plain English**: four of your six series aren't randomly wandering forever — they're actually *stable* series that experienced **one clean, sharp, one-time shift** clustering around January–March 2022. This lines up exactly with Ghana's currency depreciation and fuel-price crisis. Brent and CPI don't fit this pattern — Brent because it follows global, not domestic, dynamics; CPI because a price index mathematically can't "revert" the way a market price can.

### Differencing — turning a non-stationary series into a stationary one

**Differencing** means replacing each value with *the change from the previous value* (this month's price minus last month's price), rather than the raw price itself. This is written mathematically as `d=1` (first difference). If one differencing pass isn't enough, you can difference the differenced series again — that's `d=2` (second difference).

**Why differencing works**: if a series has a steady upward trend, the *differences* between consecutive points tend to hover around a stable average — the trend cancels out.

**What you found after first differencing (d=1)**:
- Diesel, petrol, Brent, exchange rate: **stationary** — one difference was enough.
- LPG: **borderline** — KPSS says stationary, ADF says not quite (p=0.127). You made the judgment call to treat LPG as adequately stationary at d=1, since both tests had agreed clearly at level that it was non-stationary, and documented this ambiguity honestly rather than hiding it.
- CPI: **still not stationary** even after one difference.

**Why CPI needed a second difference**: CPI is a *cumulative* index — it only ever climbs (representing the accumulation of price rises over time). Differencing once gives you "how much did inflation change this month," which can *itself* still be trending if inflation is accelerating (which Ghana's was). Differencing a second time gave you a properly stationary series (confirmed: ADF p=0.0087, KPSS p=0.10).

### Why VAR needed a different differencing rule than ARIMA

VAR treats multiple variables as one combined system, and the mathematics behind VAR requires every variable in that system to be differenced to the **same** order. Since CPI needed d=2, and CPI is part of the VAR system, **every** VAR variable — even the ones that were already fine at d=1 — got differenced to d=2 for consistency. This is a standard, safe choice: differencing a series one extra time beyond what's strictly necessary doesn't break it, it just adds a small amount of extra noise.

---

## 3. Notebook 03 — Model Identification (ACF & PACF)

### What ACF and PACF actually are

**ACF (Autocorrelation Function)**: measures how correlated a series is with a delayed ("lagged") copy of itself. Lag 1 means "compared to one month ago," lag 2 means "compared to two months ago," and so on. Each bar in an ACF plot shows this correlation at that specific lag.

**PACF (Partial Autocorrelation Function)**: similar idea, but it strips out the *indirect* influence passed through intermediate lags. For example, the raw correlation at lag 2 might partly just be an echo of the strong correlation at lag 1 — PACF removes that indirect effect and shows only the *direct* relationship at each specific lag.

**The shaded blue band** in each plot is the 95% confidence interval — bars that poke outside this band are considered statistically meaningful; bars inside it are treated as "not significantly different from zero," i.e. noise.

### Why you look at these before fitting a model

ACF and PACF patterns are a classic diagnostic tool (from Box & Jenkins' original ARIMA methodology) for guessing the right model structure *before* running an expensive search:
- A **sharp cutoff** (bars go significant then immediately drop to zero) suggests a low-order model.
- A **slow, gradual decay** suggests the series still has trend left in it (i.e., you haven't differenced enough).

### What your plots showed

For all three fuels (diesel, petrol, LPG), after first-differencing, both ACF and PACF showed a **sharp cutoff around lag 2–3**, with nothing meaningfully significant afterward. This told you two things:
1. Your differencing (d=1) was correctly chosen — there's no leftover slow decay indicating more differencing is needed.
2. The right model is probably a **low-order ARMA process** — small values for `p` (autoregressive terms) and `q` (moving-average terms), which is exactly what you searched for next.

---

## 4. Notebook 04 — ARIMA

### What ARIMA(p, d, q) actually means

ARIMA stands for **AutoRegressive Integrated Moving Average**, and it has three settings:
- **p** (AutoRegressive order): how many of the series' *own past values* are used to predict the next one. E.g. p=2 means "this month's price depends on the last 2 months' prices."
- **d** (Integrated / differencing order): how many times the series was differenced to make it stationary (this comes directly from Notebook 02).
- **q** (Moving Average order): how many of the model's *own past forecast errors* are used to correct the next prediction. This lets the model self-correct based on how wrong it recently was.

### How the right (p, d, q) was chosen

You tried every combination of p and q from 0 to 3 (with d fixed from the stationarity results) and compared them using **AIC (Akaike Information Criterion)** — a score that rewards a model for fitting the data well, but penalises it for using more parameters than necessary (to stop you from picking an overly complex model that just memorises noise). **Lower AIC = better balance of fit and simplicity.** The combination with the lowest AIC was selected.

### The train/test split

You held back the **last 6 months (Jan–Jun 2026)** as a test set the model never saw during fitting, and trained/selected orders using only the preceding **126 months**. This is standard practice: if you let the model see the data you're testing it on, you'd be measuring how well it memorised the answer, not how well it forecasts the future.

### Your results, explained

| Series | Chosen order | MAE | RMSE | MAPE | Ljung-Box p |
|---|---|---|---|---|---|
| Diesel | (0,1,3) | 0.8652 | 1.0243 | 6.17% | 0.9699 |
| Petrol | (0,1,3) | 1.4638 | 1.6534 | 9.38% | 0.9934 |
| LPG | (2,1,2) | 0.1888 | 0.2293 | 1.40% | 0.0338 |

**What each metric means**:
- **MAE (Mean Absolute Error)**: on average, how far off (in absolute GHS terms) the forecast was from the real value, ignoring direction. Easy to interpret directly in the fuel's own units.
- **RMSE (Root Mean Squared Error)**: similar, but squares the errors before averaging — this punishes large errors more heavily than small ones, so a high RMSE relative to MAE signals a few big misses rather than many small ones.
- **MAPE (Mean Absolute Percentage Error)**: the error expressed as a percentage of the actual value — useful for comparing accuracy *across* series with very different price scales (you can't directly compare a GHS 0.87 error on a ~GHS 5–17 diesel price to a GHS 1.46 error on a ~GHS 2.5–22 petrol price without converting both to percentages).

**Ljung-Box test**: checks whether the model's leftover errors (residuals) still contain a predictable pattern the model missed.
- **Null hypothesis**: residuals are "white noise" — pure randomness, no pattern left.
- **p ≥ 0.05**: good news — you failed to find evidence of leftover pattern, meaning the model captured the structure adequately.
- **p < 0.05**: bad news — there's a pattern the model didn't catch.

Diesel (p=0.97) and petrol (p=0.99) passed comfortably. **LPG did not** (p=0.034) — meaning LPG's ARIMA(2,1,2) model left some predictable structure unexplained. This became a key thread that gets resolved in the next notebook.

### The residual diagnostic plots — what each panel means

Every 4-panel diagnostic image (e.g. Figure 4.8 for diesel) shows:
1. **Standardized residual (top-left)**: the model's errors over time, rescaled so they should look like random noise bouncing around zero if the model is good. A big spike here means one specific month the model badly missed.
2. **Histogram + density (top-right)**: the *shape* of the error distribution, compared against a perfect bell curve (green line, "N(0,1)"). If your errors match a bell curve, your model's uncertainty estimates (confidence intervals) are trustworthy.
3. **Q-Q plot (bottom-left)**: another way to check for normality — if the dots follow the red diagonal line, errors are normally distributed; points curving away from the line (especially at the ends) mean "heavy tails" — more extreme values than a normal distribution would predict.
4. **Correlogram (bottom-right)**: this is the ACF of the *residuals* specifically — this is the visual companion to the Ljung-Box test. Bars poking outside the blue band mean leftover pattern.

**What you actually found**: diesel and petrol both show one huge spike (6–8 standard deviations) around late 2022 — the same event flagged throughout. This drags the histogram and Q-Q plot away from "normal," **even though** Ljung-Box passed. This is an important nuance: **Ljung-Box only checks for leftover pattern/autocorrelation — it says nothing about whether the errors are normally distributed.** A model can have zero leftover pattern (good) while still having one huge outlier that breaks the normality assumption (a separate issue). LPG's diagnostics look different: instead of one big spike, there are several moderate recurring spikes across the years, and its correlogram shows a bar poking out at lag 5 — the visual confirmation of why its Ljung-Box test failed.

**Why this matters practically**: point forecasts (the single best-guess number) remain valid even with non-normal residuals. But **prediction intervals** (the "95% confident the value will fall between X and Y" range) are built assuming normal errors — so with this outlier-driven non-normality, those intervals are less trustworthy, especially right around events like the one seen in 2022.

---

## 5. Notebook 05 — ARIMAX

### What's different from ARIMA

ARIMAX = ARIMA + eXogenous variables. It's the same modeling framework, but the model is also given **Brent crude oil prices and the USD/GHS exchange rate** as extra input variables (called "exogenous regressors" — variables the model uses to help predict, but doesn't try to forecast itself).

**The economic logic**: fuel is imported and priced internationally in USD, then converted to GHS for the domestic pump price. So it makes sense that (a) the international oil price and (b) how many GHS it takes to buy one USD should both help predict the local price.

### Your results

| Series | Order | MAE | RMSE | MAPE | Ljung-Box p |
|---|---|---|---|---|---|
| Diesel | (0,1,3) | 0.4769 | 0.5526 | 3.49% | 0.7909 |
| Petrol | (0,1,3) | 0.7162 | 0.7801 | 4.65% | 0.9618 |
| LPG | (1,1,3) | 0.1136 | 0.1349 | 0.84% | 0.4731 |

**Comparing to ARIMA**: every single metric improved. Diesel's MAE roughly halved (0.87 → 0.48). Most importantly, **LPG's Ljung-Box problem is resolved** — p went from 0.034 (failing) to 0.473 (comfortably passing). This tells you that the leftover pattern ARIMA couldn't explain in LPG's residuals was actually connected to Brent and/or exchange-rate movements — information ARIMA simply didn't have access to.

### The important caveat you must be able to explain

This result is **conditional**: when producing these forecasts, the model was fed the *actual, real* Brent and exchange-rate values from the test period. In a genuine forecasting scenario, you wouldn't know those future values in advance — you'd need to forecast them too, and any error in *those* forecasts would leak into your fuel-price forecast. So this six-month result demonstrates that Brent and the exchange rate **carry real predictive information** — it doesn't yet prove you could achieve this same accuracy in a live, fully unconditional forecasting setting.

### The residual diagnostics, revisited

The same 2022-outlier pattern persists in ARIMAX's residuals for diesel and petrol, at almost the same magnitude as plain ARIMA — meaning Brent and the exchange rate help the model's *point forecasts*, but don't fully absorb the shock's effect on the *residual distribution*. This makes sense: the shock was a **domestic** currency crisis, only partially reflected in these two specific external variables.

Interestingly, LPG's correlogram *still* shows a bar near lag 5, even though the overall Ljung-Box test now passes. This isn't a contradiction — Ljung-Box is a *joint* test across 10 lags combined; one borderline lag doesn't necessarily fail the combined test, especially once the extra regressors soak up some of the surrounding structure.

---

## 6. Notebook 06 — Rolling-Origin Forecast Evaluation

### Why a single 6-month test isn't enough

Notebooks 04 and 05 each measured accuracy on **one specific 6-month window** (Jan–Jun 2026). But what if that window happened to be unusually easy or hard? A single test could give a misleading picture. **Rolling-origin evaluation** fixes this by repeating the same experiment from many different starting points ("origins") across your history, and averaging the results — a much more robust test of "how good is this model, generally?"

### How it works, step by step

1. Start with a 96-month training window (the earliest chunk of your data).
2. Identify the best model orders (p,d,q for ARIMA/ARIMAX, lag for VAR) *once*, using only this initial window.
3. Forecast 1 to 6 months ahead from this starting point.
4. Move the "origin" forward one month, refit the *same* model orders on the now-slightly-larger training window, and forecast again.
5. Repeat this 31 times, sliding forward one month each time, until you run out of room before the end of the dataset.
6. Collect all these forecasts and compute average error metrics *per horizon* (how good is a 1-month-ahead forecast on average, vs. a 6-month-ahead forecast, etc.)

**Why the model orders are fixed rather than re-selected every time**: re-running a full AIC search 31 times for every model would be extremely slow and unnecessary for the purpose of this test — this is a standard, explicitly acknowledged simplification.

### What "MAE at horizon 3" means

If a metric is reported "at horizon 3," it means: across all 31 origins, how far off (on average) was the forecast made **3 months in advance**? Naturally, forecasting further into the future is harder, so error metrics should generally get worse (bigger) as the horizon number increases — and your Figure 4.19 plot confirms this pattern clearly for every model and fuel.

### The key finding: rolling results tell a different story than the single 6-month test

In the single 6-month holdout test (Notebooks 04–05), ARIMAX beat ARIMA. But in the rolling-origin evaluation across 31 repeated tests, **ARIMA is generally more accurate than ARIMAX for diesel and petrol at every horizon.** These aren't contradictory findings — they're answering different questions:
- The single holdout tells you: "how did each model do in this one specific recent period?"
- The rolling evaluation tells you: "on average, across many different historical starting points, which model tends to do better?"

Both are legitimate and worth reporting — they just serve different purposes.

### The Diebold-Mariano (DM) test — explained properly

Just because Model A has a lower MAE than Model B doesn't automatically mean Model A is *genuinely* better — the difference might just be random luck in this particular sample. The **Diebold-Mariano test** answers: "is the difference in forecasting accuracy between these two models statistically real, or could it just be noise?"

- **Null hypothesis**: the two models have *equal* predictive accuracy.
- **DM statistic**: negative means the first model named has *smaller* errors (i.e., is more accurate) than the second.
- **p < 0.05**: reject the null — the accuracy difference is statistically real, not just random chance.

**Your key results** (at the 1-month horizon):
- **Diesel**: ARIMA significantly beats VAR (p=0.024). ARIMA vs. ARIMAX isn't significant (p=0.35) — meaning even though ARIMA had lower error numerically, you can't be statistically confident that's a real, generalisable advantage.
- **Petrol**: ARIMA significantly beats *both* ARIMAX (p=0.012) and VAR (p=0.007).
- **LPG**: **None of the three comparisons are statistically significant** (p=0.73, 0.18, 0.28). This is an important, honest finding: even though ARIMA has the lowest numerical MAE for LPG, the statistical test says you can't confidently claim any one model is truly better than another for LPG — they're statistically indistinguishable given the amount of data available.

**Why this distinction matters for your defense**: it shows methodological maturity to say "ARIMA numerically wins for LPG, but the difference isn't statistically significant" rather than just reporting the lowest number as if it were automatically the true best model.

---

## 7. Notebook 07 — VAR (Vector Autoregression)

### What VAR is, and how it's different from everything before it

ARIMA and ARIMAX are **univariate** (or conditionally univariate) — they forecast one series at a time (fuel price), optionally using other variables as one-directional helpers. **VAR treats every variable as equally important and lets them all influence each other simultaneously.** In a VAR model, diesel's future doesn't just depend on diesel's past — it also depends on petrol's past, LPG's past, Brent's past, the exchange rate's past, and CPI's past, *and* those other variables' futures depend on diesel's past too. It's a fully interconnected system, not a one-way street.

### Why VAR uses second differences

As explained in Notebook 02: because CPI needs d=2 to become stationary, and VAR requires every variable in the system to share the same differencing order, all six variables (diesel, petrol, LPG, Brent, exchange rate, CPI) were differenced twice before fitting VAR.

### Reversing the differencing (integration) to get real forecasts back

A VAR fitted on twice-differenced data produces forecasts *in twice-differenced units* — which aren't directly meaningful as prices. To convert back:
1. Take the last known *actual first difference* (Δy) and add each forecasted second-difference (Δ²y), accumulated step by step, to rebuild a forecast of the first differences.
2. Take the last known *actual price level* and add each of those reconstructed first-difference forecasts, accumulated step by step, to rebuild a forecast of the actual price level.

This is exactly what the code's `cumsum()` (cumulative sum) operations do — this two-step "undo the differencing" process is called **integration**, and it's the "I" that would appear if this were called "VARIMA."

### Choosing the lag order — and the important judgment call you made

VAR needs a **lag order** setting (how many past months of *every* variable to include) — the VAR equivalent of ARIMA's `p`. You tested lags 1 through 6, checked each for **stability** (a mathematical requirement that a VAR's dynamics don't spiral off to infinity — without this check, a VAR could produce runaway forecasts), and compared AIC and BIC among the stable candidates.

**AIC favoured lag 6. BIC favoured lag 2.** BIC penalises extra parameters more heavily than AIC does, so this disagreement is a red flag: with 6 variables and lag 6, the model needs to estimate 37 parameters *per equation* (1 intercept + 6 variables × 6 lags), off a training sample of roughly only 118 usable data points — a thin ratio that risks the model fitting noise rather than real structure ("overparameterization").

**You tested this concern directly** rather than just trusting AIC blindly: you fit both lag 2 and lag 6, and compared their residual diagnostics and actual forecast accuracy. The result was surprising — reducing to lag 2 did **not** fix the residual problems (normality actually got *worse*, not better), and lag 2 produced *worse* forecast accuracy for petrol and LPG. This told you the residual issues aren't really about too many parameters — they're almost certainly the same 2022 structural shock showing up again, which a shorter-memory model (lag 2) captures even less completely than the longer one. **This is exactly the kind of evidence-based defense a supervisor or examiner wants to see** — not "AIC said so," but "we checked the alternative directly and it didn't help."

### VAR forecast results and what they mean

| Series | MAE | RMSE |
|---|---|---|
| Diesel | 2.05 | 2.17 |
| Petrol | 2.22 | 2.34 |
| LPG | 0.22 | 0.28 |

Compared to ARIMA/ARIMAX, VAR's errors for diesel and petrol are noticeably **worse** — VAR is not your best point-forecasting tool for these two fuels. This isn't a failure of the model — VAR wasn't built to win at point forecasting; it was built to reveal *how the variables interact*, which ARIMA and ARIMAX structurally cannot show you at all.

### Residual whiteness and normality tests (the multivariate versions)

These are the VAR equivalents of Ljung-Box and the normality checks, but done **jointly across the whole system** rather than one equation at a time.
- **Whiteness test**: null hypothesis is "no leftover autocorrelation anywhere in the system's residuals." Your test rejected this (p<0.001) — some structure remains unexplained.
- **Normality test**: null hypothesis is "residuals are normally distributed." Also rejected (p<0.001).

Both failures persisted even at the more conservative lag 2, reinforcing that this is about the 2022 shock, not model complexity.

### Granger causality — what it actually tests (and what it doesn't)

**Granger causality is not the same thing as true cause-and-effect.** It only tests: "does knowing variable X's past values help predict variable Y, above and beyond what Y's own past already tells you?" If yes, X is said to "Granger-cause" Y — but this is a statement about *predictive usefulness in this specific fitted model*, not a proof of a real-world causal mechanism.

**Your key results**: the exchange rate Granger-causes all three fuel prices (all p<0.001) — the strongest, most consistent relationship in the whole table. Brent Granger-causes diesel but not petrol or LPG. CPI doesn't Granger-cause any of the three fuels directly.

### Impulse Response Functions (IRFs) — what the graphs show

An IRF answers: "if variable X gets hit with a one-time surprise shock today, how does variable Y respond over the following months?" Each subplot in Figure 4.26 traces out this response over a 12-month horizon, with a shaded band showing the uncertainty around that estimated response (calculated using Monte Carlo simulation — repeatedly resampling and refitting to see the range of plausible response shapes).

**What you found**: diesel and petrol both respond visibly to exchange-rate and Brent shocks, oscillating up and down over the following months before settling — the exchange-rate response is clearly larger in size than the Brent response, matching the stronger Granger-causality finding above. LPG's responses are smaller and their uncertainty bands are wide enough that you can't confidently say the response differs from zero.

### Forecast-Error Variance Decomposition (FEVD) — the most surprising result in your whole study

FEVD answers a different question from Granger causality: "of all the uncertainty in my 6-month-ahead forecast for fuel price Y, what percentage of that uncertainty comes from each variable in the system?"

**What you found**: **CPI accounts for 65–68% of the forecast-error variance for all three fuel prices** — far more than each fuel's own past shocks (16–23%) or Brent/the exchange rate (8% or less each).

**Why this seems to contradict the Granger-causality table** (CPI didn't Granger-cause any fuel directly) **— and why it actually doesn't**: Granger causality only measures a *direct* one-to-one link. FEVD measures *total* influence flowing through the *entire system*, including indirect paths — CPI influencing the exchange rate, which influences fuel prices, for example. Since CPI is strongly correlated with everything else in your system (Section 4.1) and the VAR links all six variables together dynamically, CPI's influence shows up strongly in the total system-wide accounting even without a direct one-step link. The takeaway: CPI functions as a broad signal of the same underlying inflation/currency pressures driving fuel prices, transmitted indirectly rather than directly.

---

## 8. How Chapter 4 ties all of this together

Chapter 4 doesn't introduce new analysis — every table and figure in it is a direct, labelled excerpt from these seven notebooks, arranged into the thesis narrative:
- **4.1–4.2** = Notebook 01's output.
- **4.2 (stationarity)** = Notebook 02's output.
- **4.3–4.4** = Notebooks 03, 04, 05's output (model identification, ARIMA, ARIMAX).
- **4.5** = Notebook 06's output (rolling-origin, Diebold-Mariano).
- **4.6** = Notebook 07's output (VAR, lag robustness check, Granger causality, IRF, FEVD).
- **4.7–4.9** = your own interpretation, tying every result back to the five research objectives from Chapter 1.

---

## 9. If your supervisor asks you to summarise the whole study in two minutes

A short version you could say out loud:

> "I tested three approaches to forecasting Ghana's fuel prices. First, ARIMA — using only each fuel's own history — which worked reasonably well but struggled with LPG specifically. Second, ARIMAX — adding Brent crude and the exchange rate — which improved accuracy for all three fuels and fixed LPG's specific problem, though that result assumes we already know those external values in advance. Third, VAR — treating all six variables as one connected system — which wasn't the most accurate at pure point forecasting, but revealed that the exchange rate is the strongest driver across all three fuels, and that CPI has a surprisingly large *indirect* influence on fuel-price forecast uncertainty. Along the way, I found and explicitly tested for a structural break around the 2022 currency crisis, which explains most of the non-normal, outlier-driven behaviour I see in every model's residuals — including double-checking that a simpler VAR specification wouldn't have fixed that problem, which it didn't."

---

## 10. Quick-reference glossary

| Term | One-line meaning |
|---|---|
| Stationary | Statistical properties (mean, variance) don't change over time |
| Differencing (d) | Replacing values with period-to-period changes, to remove trend |
| ADF test | Null: non-stationary. Low p-value = evidence of stationarity |
| KPSS test | Null: stationary. Low p-value = evidence of non-stationarity |
| Zivot-Andrews test | Like ADF, but allows for one structural break |
| ACF / PACF | How correlated a series is with its own past, direct vs. total |
| AIC / BIC | Model-selection scores; lower is better; BIC penalises complexity more |
| ARIMA(p,d,q) | Forecasts using own past values (p), own past errors (q), after differencing (d) |
| ARIMAX | ARIMA plus external predictor variables |
| MAE / RMSE / MAPE | Average error size — absolute, squared-then-averaged, percentage |
| Ljung-Box test | Null: no leftover pattern in residuals |
| Rolling-origin evaluation | Repeating the forecast test from many different starting points |
| Diebold-Mariano test | Tests whether one model's accuracy is *statistically* better than another's |
| VAR | Models multiple variables jointly, each depending on all their pasts |
| Granger causality | Does X's past help predict Y, beyond Y's own past? (not true causation) |
| Impulse response function | How does Y react over time to a one-time shock in X? |
| Forecast-error variance decomposition | What share of Y's forecast uncertainty comes from each variable? |
