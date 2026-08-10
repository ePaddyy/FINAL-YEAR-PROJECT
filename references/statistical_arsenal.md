# Statistical Arsenal — Ghana Fuel Price Forecasting Project

> **Your complete guide to every statistical tool, test, and model used in this project.**  
> Study this document. Every item here will appear in Chapters 3 and 4.

---

## Overview

| Phase | What You Are Doing |
|---|---|
| Phase 1 | Explore and understand the data |
| Phase 2 | Test and prepare the data for modelling |
| Phase 3 | Identify the right model structure |
| Phase 4 | Build the models |
| Phase 5 | Validate the models |
| Phase 6 | Forecast and evaluate accuracy |
| Phase 7 | Interpret findings for policy |

---

## Phase 1 — Exploratory Analysis

### Descriptive Statistics
Compute mean, standard deviation, minimum, maximum, and skewness for all price series.  
**Why:** Tells you the basic behaviour of your data before any modelling begins. Always the first step.

### Time Series Plots
Visual inspection of trends, seasonality, and structural breaks across the full study period.  
**Why:** If you cannot see a pattern visually, no model will reliably find it either.

### Correlation Analysis
Measures the linear relationship between fuel prices and external variables (exchange rate, Brent crude).  
**Why:** Tells you which external variables are worth including before you build a model.

---

## Phase 2 — Stationarity Testing

> **Key concept:** ARIMA and VAR models require *stationary* data — meaning the mean and variance of the series do not change over time. Fuel prices are almost certainly non-stationary because they trend upward over time. You must test for this and correct it before modelling.

### Augmented Dickey-Fuller (ADF) Test
The standard test for stationarity.

- **Null hypothesis:** The series has a unit root (non-stationary)
- **Decision rule:** If p-value < 0.05 → reject null → series is stationary
- **If non-stationary:** Difference the series and retest

### KPSS Test
Used alongside ADF for confirmation. Tests the *opposite* hypothesis to ADF.

- **Null hypothesis:** The series is stationary
- **Decision rule:** If p-value < 0.05 → reject null → series is non-stationary
- **Why use both:** ADF and KPSS together give stronger evidence than either alone

### Zivot-Andrews Test
An advanced stationarity test that accounts for **structural breaks** — sudden shifts caused by events like the 2020 COVID crash or the 2022 Russia-Ukraine war.

- **Why it matters for Ghana:** Standard ADF ignores structural breaks and can give misleading results in volatile economies. Zivot-Andrews finds the break point automatically.
- **Output:** Date of the structural break + whether the series is stationary accounting for it

### Differencing
If a series is non-stationary, subtract each value from the previous one. This removes trends and makes the series stationary.

- The number of times you difference = **order of integration**, denoted *d* in ARIMA(p, d, q)
- First difference: Δy_t = y_t − y_{t−1}
- Most fuel price series require only first differencing (d = 1)

---

## Phase 3 — Model Identification

### ACF — Autocorrelation Function
Shows how correlated a series is with its own past values at different lags.  
**Used for:** Identifying the MA(q) component of ARIMA.

### PACF — Partial Autocorrelation Function
Shows the *direct* correlation between a series and its lagged values, removing the effect of intermediate lags.  
**Used for:** Identifying the AR(p) component of ARIMA.

### Information Criteria — AIC and BIC
When you have multiple candidate models, these criteria help you choose the best one. **Lower values are better.**

| Criterion | Full Name | Characteristic |
|---|---|---|
| AIC | Akaike Information Criterion | Rewards fit, mild complexity penalty |
| BIC | Bayesian Information Criterion | Heavier complexity penalty than AIC |

**Rule of thumb:** Use BIC when you want a more parsimonious (simpler) model.

### Lag Selection for VAR
For VAR models, use AIC, BIC, HQ (Hannan-Quinn), and FPE criteria together to determine how many lags to include.  
**In Python:** `statsmodels` VAR model has a `.select_order()` method that computes all four automatically.

---

## Phase 4 — Model Estimation

### ARIMA(p, d, q) — Baseline Model

| Parameter | Meaning |
|---|---|
| p | Number of autoregressive (AR) terms |
| d | Degree of differencing |
| q | Number of moving average (MA) terms |

- **What it does:** Forecasts future prices using only the *past values* of the price series itself
- **Limitation:** Ignores external factors — cannot respond to cedi depreciation or crude oil spikes
- **Role in your project:** Baseline model. All other models are compared against it.

### ARIMAX(p, d, q) — External Shock Model

- Same structure as ARIMA but adds **external regressors** as inputs
- Your external variables: exchange rate (GHS/USD) and Brent crude oil prices
- The X variables are treated as fixed inputs — they influence fuel prices but are not jointly modelled
- **Advantage over ARIMA:** Captures the impact of external shocks on fuel prices
- **Role in your project:** Middle model. Tests whether external shocks improve forecast accuracy.

### VAR(p) — System Model

- Models **all variables simultaneously** in a system
- Every variable is regressed on lagged values of all other variables
- Variables included: petrol/diesel/LPG prices, exchange rate, Brent crude, and CPI
- Captures **bidirectional** relationships — e.g. fuel prices feeding into inflation and vice versa
- **Advantage over ARIMAX:** CPI can be included without endogeneity problems because all variables are treated symmetrically
- **Role in your project:** Most complex model. Tests whether system-wide dynamics improve forecasting.

> **Note on CPI:** CPI is included *only* in VAR, not in ARIMAX. This is because CPI is endogenous — fuel prices affect inflation which feeds back into fuel prices. ARIMAX cannot handle this feedback loop; VAR can.

---

## Phase 5 — Model Diagnostics

### Residual Analysis
After fitting a model, the residuals (errors) should behave like **white noise** — no patterns, no autocorrelation, no trends.  
**If patterns remain:** The model has not captured all the information in the data and needs revision.

### Ljung-Box Test
Formally tests whether residuals are white noise.

- **Null hypothesis:** Residuals are independently distributed (white noise)
- **Decision rule:** If p-value > 0.05 → residuals are white noise → model is adequate

### Jarque-Bera Test
Tests whether residuals are **normally distributed**.  
**Why it matters:** Valid statistical inference assumes normally distributed errors.

### ARCH Test (Engle's Test)
Tests for **volatility clustering** in residuals — periods of high volatility followed by more high volatility.  
**Why it matters:** Common in fuel price data. If present, a GARCH extension may be needed.

### Granger Causality Test
Tests whether past values of one variable help predict another variable *beyond what its own history tells you.*

**Example question:** Does knowing past exchange rates improve our forecast of petrol prices beyond what petrol price history alone tells us?

- **Null hypothesis:** Variable X does NOT Granger-cause Variable Y
- **Decision rule:** If p-value < 0.05 → X Granger-causes Y → X belongs in your model
- **Important:** Granger causality is about *predictive* causality, not true causal relationships

---

## Phase 6 — Forecasting and Evaluation

### Train/Test Split
Split your data into two parts:

| Set | Period (example) | Purpose |
|---|---|---|
| Training set | 2013–2021 | Build and fit the model |
| Test set | 2022–2023 | Evaluate on unseen data |

**Why:** A model that fits training data perfectly can still fail on new data (overfitting). The test set simulates real-world forecasting.

### Forecast Horizons: 1–6 Months
You will produce forecasts at multiple time steps ahead:

- **1 month:** Short-term, highest accuracy expected
- **3 months:** Medium-term, one quarter forward
- **6 months:** Long-term, accuracy will degrade

**Why multiple horizons matter:** Different models deteriorate at different rates. ARIMA is generally good at 1–2 months but weakens quickly. ARIMAX and VAR may hold up better at longer horizons because they use external information. This comparison *across* horizons is a key contribution of your project.

### RMSE — Root Mean Square Error
$$RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}$$

- Penalises **large errors heavily** due to squaring
- Most commonly reported metric in forecasting literature
- Unit: Same as the original variable (GHS per litre)

### MAE — Mean Absolute Error
$$MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|$$

- Average size of errors regardless of direction
- More **interpretable** than RMSE — directly tells you average error in GHS
- Less sensitive to outliers than RMSE

### MAPE — Mean Absolute Percentage Error
$$MAPE = \frac{1}{n}\sum_{t=1}^{n}\left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100$$

- Expresses error as a **percentage** of actual values
- Most useful for communicating accuracy to non-technical audiences like policymakers
- Example: "The model forecasts petrol prices with an average error of 4.2%"

### Diebold-Mariano (DM) Test
When you compare RMSE across models, ARIMAX may show lower RMSE than ARIMA. But is that difference **statistically significant**, or could it have happened by chance?

The Diebold-Mariano test formally answers this question.

- **Null hypothesis:** Both models have equal forecast accuracy
- **Alternative hypothesis:** One model is significantly more accurate
- **Decision rule:** If p-value < 0.05 → the difference in accuracy is statistically significant

**Why this matters:**  
Without DM test → your comparison is descriptive and easy to challenge.  
With DM test → your comparison is statistically rigorous and hard to challenge.

> **In Python:** `statsmodels.stats.stattools.diebold_mariano` — one function call after you have your forecast errors.

---

## Phase 7 — VAR-Specific Interpretation Tools

### Impulse Response Functions (IRF)
Shows how a **shock** to one variable propagates through all other variables over time.

**Example:** What happens to petrol prices over the next 6 months if Brent crude suddenly increases by $10?

- X-axis: Time periods after the shock
- Y-axis: Response of the target variable
- Confidence bands show uncertainty

**Why it is powerful:** Visually demonstrates the real-world impact of external shocks — directly relevant to your research question about advance warning for policymakers.

### Forecast Error Variance Decomposition (FEVD)
Tells you what **percentage** of the forecast error in fuel prices is explained by each variable in the system.

**Example output:**
> "After 6 months, 38% of petrol price forecast error is attributable to exchange rate shocks, 29% to Brent crude shocks, and 18% to own-price shocks."

**Why it matters:** This is one of the strongest policy findings your project can produce — it tells policymakers *which external factor* is driving forecast uncertainty the most.

---

## Complete Summary Table

| Phase | Tool | Purpose | Chapter |
|---|---|---|---|
| Exploration | Descriptive statistics | Understand data behaviour | 4 |
| Exploration | Time series plots | Visual pattern detection | 4 |
| Exploration | Correlation analysis | Variable selection | 4 |
| Stationarity | ADF Test | Test for unit root | 3 & 4 |
| Stationarity | KPSS Test | Confirm stationarity | 3 & 4 |
| Stationarity | Zivot-Andrews Test | Account for structural breaks | 3 & 4 |
| Identification | ACF / PACF | Determine ARIMA orders | 3 & 4 |
| Identification | AIC / BIC | Model selection criteria | 3 & 4 |
| Estimation | ARIMA | Univariate baseline forecast | 3 & 4 |
| Estimation | ARIMAX | External shock model | 3 & 4 |
| Estimation | VAR | Full system model | 3 & 4 |
| Diagnostics | Ljung-Box Test | Residual white noise check | 4 |
| Diagnostics | Jarque-Bera Test | Residual normality check | 4 |
| Diagnostics | ARCH Test | Volatility clustering check | 4 |
| Diagnostics | Granger Causality | Variable inclusion justification | 4 |
| Forecasting | Train/Test Split | Out-of-sample evaluation | 3 & 4 |
| Evaluation | RMSE / MAE / MAPE | Forecast accuracy metrics | 4 |
| Evaluation | Diebold-Mariano Test | Statistical model comparison | 4 |
| Interpretation | IRF | Shock propagation analysis | 4 |
| Interpretation | FEVD | Variance attribution by variable | 4 |

---

*Ghana Fuel Price Forecasting Project — Department of Statistics and Actuarial Science, University of Ghana*
