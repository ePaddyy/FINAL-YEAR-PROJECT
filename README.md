FINAL-YEAR-PROJECT

TIME SERIES ANALYSIS ON FUEL PRICE


# Fuel Price Time-Series Analysis

## 1. Purpose

This project studies and forecasts monthly fuel prices in Ghana using historical fuel prices and selected economic variables. The main fuel-price series are:

- Diesel price in GHC per litre
- Petrol price in GHS per litre
- LPG price in GHS per kilogram

The external variables are:

- Brent crude oil price in USD per barrel
- USD/GHS exchange rate
- Consumer Price Index (CPI)

The analysis is organized as a sequence of notebooks. Each notebook answers a different question:

1. What does the cleaned dataset look like?
2. Are the series statistically suitable for time-series modeling?
3. What ARIMA orders appear reasonable from the correlation structure of the differenced data?
4. How well can each fuel price be forecast using its own past values?
5. Does adding economic variables improve the forecasts?
6. Which forecasting approach performs best on unseen observations?
7. What happens when all fuel and macroeconomic variables are modeled together as a system?

The source data used by the modeling notebooks is `data/dataset_final/full_data.csv`.

## 2. Data and Time Index

The dataset contains 132 monthly observations from July 2015 through June 2026. It contains seven columns:

- `month`: original month label, such as `2015M07`
- Three fuel-price columns
- Three external-variable columns

The notebooks convert `month` to a pandas datetime index using the format `%YM%m`. The resulting index uses monthly-start frequency (`MS`). This matters because time-series models need the observations in their true chronological order and need to know that the interval between observations is one month.

The final cleaned data has:

- 132 observations
- 6 numeric variables
- No missing values in the saved dataset

The first observations are from July 2015. The last observations are from June 2026.

## 3. Notebook 01: Data Cleaning and EDA

File: `notebooks/01_data_cleaning_eda.ipynb`

### What was done

The notebook:

1. Loaded `full_data.csv`.
2. Inspected the first rows, data types, dimensions, and summary statistics.
3. Converted the monthly string into a datetime column.
4. Set the datetime column as the index.
5. Declared the index frequency as monthly start.
6. Checked missing values.
7. Examined correlations, distributions, trends, pairwise relationships, and possible outliers.

### Why these steps were used

Time-series models depend on correctly ordered observations. Converting the month label into a datetime index prevents the months from being treated as ordinary text and allows pandas and statsmodels to handle monthly operations correctly.

Checking missing values is important because many time-series tests and models cannot work reliably when observations are absent. The saved output shows zero missing values in every column, so no imputation was required.

The correlation matrix and plots provide an initial view of how the variables move together. They do not prove causation, but they help motivate multivariate models and identify relationships worth testing more formally.

### Main EDA findings

The descriptive output shows substantial variation across the study period:

- Diesel ranged from 2.75 to 17.43 GHC/litre.
- Petrol ranged from 2.65 to 22.58 GHS/litre.
- LPG ranged from 3.35 to 15.00 GHS/kg.
- Brent crude ranged from 26.60 to 117.30 USD/barrel.
- USD/GHS ranged from 3.80 to 15.99.
- CPI ranged from 55.00 to 270.80.

The fuel prices are strongly correlated in levels:

| Relationship       | Correlation |
| ------------------ | ----------: |
| Diesel and petrol  |      0.9909 |
| Diesel and LPG     |      0.9607 |
| Petrol and LPG     |      0.9274 |
| Diesel and USD/GHS |      0.9551 |
| LPG and USD/GHS    |      0.9767 |
| Diesel and CPI     |      0.9261 |
| Petrol and Brent   |      0.6883 |

These strong level correlations suggest that the fuel prices may share common trends. They also warn us not to interpret simple level correlations as independent effects, because trending variables can appear highly correlated even when their short-run relationship is weaker.

## 4. Notebook 02: Stationarity Testing

File: `notebooks/02_stationarity_testing.ipynb`

### Why stationarity matters

A weakly stationary series has a stable statistical behavior over time, particularly a stable mean and variance. Standard ARIMA and VAR modeling assumptions are easier to justify when the modeled series is stationary.

If a non-stationary series is modeled directly in levels, the model may produce misleading relationships or unstable forecasts. Differencing removes changes in the level or trend:

- First difference: $\Delta y_t = y_t - y_{t-1}$
- Second difference: $\Delta^2 y_t = \Delta y_t - \Delta y_{t-1}$

### Tests used

Three tests were used at levels:

- **ADF test:** null hypothesis is a unit root, or non-stationarity. A small p-value supports stationarity.
- **KPSS test:** null hypothesis is stationarity. A small p-value supports non-stationarity.
- **Zivot-Andrews test:** allows for one possible structural break when testing for a unit root.

Using both ADF and KPSS is useful because their null hypotheses point in opposite directions. Agreement between them is more convincing than relying on one test alone. Zivot-Andrews is useful here because fuel and macroeconomic series may contain a major break around the 2021-2022 period.

### Results at levels

ADF and KPSS generally identify all six level series as non-stationary. The Zivot-Andrews test detects possible breaks for several variables:

| Series  | ADF conclusion | KPSS conclusion | Zivot-Andrews result  | Estimated break |
| ------- | -------------- | --------------- | --------------------- | --------------- |
| Diesel  | Non-stationary | Non-stationary  | Stationary with break | 2022-01         |
| LPG     | Non-stationary | Non-stationary  | Stationary with break | 2022-03         |
| Petrol  | Non-stationary | Non-stationary  | Stationary with break | 2022-02         |
| Brent   | Non-stationary | Non-stationary  | Non-stationary        | 2021-01         |
| USD/GHS | Non-stationary | Non-stationary  | Stationary with break | 2022-03         |
| CPI     | Non-stationary | Non-stationary  | Non-stationary        | 2022-02         |

The break dates are evidence that the 2021-2022 period deserves attention. They do not automatically prove that a single structural-break model is required, but they explain why the level series should not be treated as stable throughout the sample.

### Results after first differencing

After first differencing:

- Diesel: ADF and KPSS both support stationarity.
- Petrol: ADF and KPSS both support stationarity.
- Brent: ADF and KPSS both support stationarity.
- USD/GHS: ADF and KPSS both support stationarity.
- LPG: KPSS supports stationarity, but ADF does not reject non-stationarity at the 5% level.
- CPI: both tests still indicate non-stationarity.

CPI becomes stationary after second differencing. The second-difference test reports an ADF p-value of 0.0087 and a KPSS p-value of 0.10.

### Modeling implication

The current ARIMA, ARIMAX, and identification notebooks use $d=1$ as a practical common differencing order for the fuel-price modeling workflow. This is strongly supported for diesel and petrol, mostly supported for the external variables, and less decisive for LPG and CPI.

This is an important limitation rather than something to hide. A more refined final analysis should consider:

- using $d=2$ for CPI in models where CPI is directly modeled;
- testing whether LPG needs a second difference or a break adjustment;
- considering a structural-break or intervention variable for 2021-2022;
- checking whether the variables are cointegrated before deciding between differenced models and a VECM.

The current notebooks are forecasting notebooks, not a complete cointegration analysis.

## 5. Notebook 03: Model Identification

File: `notebooks/03_model_identification.ipynb`

### What was done

The notebook creates first differences for the three fuel prices and the three external variables. It then plots:

- ACF: autocorrelation function
- PACF: partial autocorrelation function

The plots are produced for the three fuel-price series with up to 24 monthly lags.

### Why ACF and PACF were used

The ACF shows how a series is related to its own previous values. The PACF shows the relationship at a particular lag after accounting for shorter lags.

They provide starting information for selecting ARIMA orders:

- The AR order $p$ is often informed by the PACF.
- The MA order $q$ is often informed by the ACF.

These plots are not an automatic proof of the correct order. That is why the next notebooks compare a small grid of candidate models using AIC and then evaluate forecasts on observations not used for fitting.

## 6. Notebook 04: ARIMA Modeling

File: `notebooks/04_arima_modeling.ipynb`

### What ARIMA means

ARIMA models a series using:

- **AR ($p$):** dependence on previous observations;
- **I ($d$):** differencing needed to remove non-stationarity;
- **MA ($q$):** dependence on previous forecast errors.

An ARIMA model is appropriate as a baseline because it uses only the historical behavior of the target fuel price. It answers: how predictable is the series from its own past?

### Implementation

The notebook:

1. Reserves the final six months as a test period.
2. Uses the earlier observations as the training period.
3. Tests all combinations where $p$ and $q$ range from 0 to 3, with $d=1$.
4. Selects the order with the lowest training-set AIC for each fuel series.
5. Fits the selected model to the training data.
6. Forecasts the six-month holdout period.
7. Calculates MAE, RMSE, and MAPE.

The selected orders were:

| Series | Selected ARIMA order |
| ------ | -------------------- |
| Diesel | (0, 1, 3)            |
| Petrol | (0, 1, 3)            |
| LPG    | (2, 1, 2)            |

### Why AIC was used

AIC balances goodness of fit against model complexity. A model with more parameters can fit the training data better simply because it is more flexible. AIC penalizes that extra complexity. The selected order is therefore a reasonable parsimonious candidate, not merely the most complicated model that fits the sample.

### Holdout results

| Series |    MAE |   RMSE |
| ------ | -----: | -----: |
| Diesel | 0.8652 | 1.0243 |
| Petrol | 1.4638 | 1.6534 |
| LPG    | 0.1888 | 0.2293 |

The metrics are in the original units of each series. Lower values indicate smaller forecast errors. Comparisons between different fuel types should be made carefully because the series have different scales.

### Residual diagnostics

The same Ljung-Box check at lag 10 is applied to every fitted fuel model. Diesel and petrol ARIMA residuals do not show evidence of remaining autocorrelation, while LPG ARIMA does (p-value approximately 0.034). This means the LPG ARIMA model needs caution even though its forecast metrics are reported.

## 7. Notebook 05: ARIMAX Modeling

File: `notebooks/05_arimax_modeling.ipynb`

### What ARIMAX adds

ARIMAX extends ARIMA by adding exogenous predictors, also called external regressors. Here, the external variables are Brent crude price and USD/GHS exchange rate, as specified in the project methodology.

The motivation is economic as well as statistical:

- Brent represents the international crude-oil market.
- USD/GHS represents the exchange-rate channel through which imported fuel costs can change in Ghanaian currency.

ARIMAX answers: can fuel prices be forecast more accurately when relevant economic information is available?

### Implementation

The notebook uses the same six-month holdout and the same candidate order grid as ARIMA. It fits the external variables during training and supplies the corresponding test-period external values when forecasting.

This setup is a conditional forecast: it assumes that future Brent and USD/GHS values are known or are available from another forecasting process. In a real deployment, those external variables would themselves need to be forecast, and uncertainty in those forecasts would pass into the fuel-price forecast.

The selected orders were:

| Series | Selected ARIMAX order |
| ------ | --------------------- |
| Diesel | (0, 1, 3)             |
| Petrol | (0, 1, 3)             |
| LPG    | (1, 1, 3)             |

### Holdout results

| Series |    MAE |   RMSE |
| ------ | -----: | -----: |
| Diesel | 0.4769 | 0.5526 |
| Petrol | 0.7162 | 0.7801 |
| LPG    | 0.1136 | 0.1349 |

ARIMAX improves the LPG forecast substantially relative to ARIMA in this holdout. It has a slightly higher diesel RMSE than ARIMA, so adding predictors does not automatically improve every series.

The ARIMAX residual diagnostics pass the lag-10 Ljung-Box check for all three series. In particular, the LPG p-value increases to approximately 0.473, so adding Brent and exchange-rate information removes the residual autocorrelation warning seen in the LPG ARIMA model.

## 8. Notebook 06: Forecast Evaluation

File: `notebooks/06_forecast_evaluation.ipynb`

### Why a benchmark is necessary

A sophisticated model should be compared with a simple rule. Otherwise, it is impossible to know whether the additional modeling complexity provides useful predictive value.

The notebook compares:

- ARIMA
- ARIMAX
- VAR

### Metrics

- **MAE:** average absolute error. It is easy to interpret in the original units.
- **RMSE:** square root of average squared error. It penalizes large errors more heavily than MAE.
- **MAPE:** mean absolute percentage error relative to the observed value.
- **Diebold-Mariano:** tests whether two models have significantly different forecast losses.

The notebook uses rolling-origin forecasts at horizons of 1, 2, 3, 4, 5, and 6 months. There are 31 forecast origins for each horizon, and each forecast uses only the data available at that origin.

In addition to the individual horizon results, the notebook reports cumulative results pooling horizons 1, 3, and 6 months. It also reports Diebold-Mariano comparisons for each of those cumulative cutoffs and for each model pair: ARIMA versus ARIMAX, ARIMA versus VAR, and ARIMAX versus VAR.

### Current interpretation

The rolling results generally favor ARIMA for diesel and petrol. LPG errors are closer between ARIMA and ARIMAX at short horizons. The Diebold-Mariano table provides formal pairwise comparisons using squared-error loss.

## 9. Notebook 07: VAR Modeling

File: `notebooks/07_var_modeling.ipynb`

### What VAR does

A vector autoregression models several time series together. Each variable is explained by its own past values and the past values of the other variables in the system.

This is appropriate for the project because fuel prices, crude oil prices, exchange rates, and inflation may influence one another over time. VAR is therefore a useful system-level comparison against the single-series ARIMA and ARIMAX models.

### Implementation

The notebook:

1. Combines the three fuel prices and three external variables.
2. Second-differences the complete system because CPI remains non-stationary after first differencing.
3. Searches lag lengths from 1 through 6 months.
4. Checks whether each fitted VAR is stable.
5. Selects the lowest-AIC stable candidate.
6. Forecasts the second-differenced system for six months.
7. Integrates the forecasts twice to reconstruct the original price levels.
8. Evaluates the three fuel-price forecasts using MAE, RMSE, and MAPE.

The notebook also performs the system analyses required for the VAR objective:

- multivariate residual whiteness testing;
- multivariate residual normality testing;
- pairwise Granger-causality tests for each fuel target;
- impulse-response analysis over 12 steps;
- forecast-error variance decomposition over six steps.

Stability is checked because a low AIC is not enough if the dynamic system produces explosive or unreliable forecasts. In the current six-month run, the notebook selected the best stable candidate, lag 6.

### Current VAR results

| Series |    MAE |   RMSE |
| ------ | -----: | -----: |
| Diesel | 2.0456 | 2.1683 |
| Petrol | 2.2219 | 2.3378 |
| LPG    | 0.2198 | 0.2757 |

The current VAR is less accurate than the best univariate or ARIMAX approach for diesel and petrol in the six-month holdout. That does not make VAR useless: it answers a different question about joint dynamics. However, the result suggests that the extra system complexity does not improve this particular out-of-sample forecast period.

The current residual diagnostics reject both the VAR whiteness and normality null hypotheses at the 5% level. The VAR should therefore be interpreted as a useful system-analysis benchmark, but not as a fully adequate residual model without further refinement.

### Differencing consistency warning

The second-difference choice is consistent with the stationarity tests, including the CPI result. A VECM remains a possible extension because differencing can remove long-run information.

## 10. Overall Interpretation

The analysis supports several cautious conclusions:

1. Fuel prices and the macroeconomic variables have strong common movement in levels.
2. The level series are generally non-stationary, so differencing is necessary for the current ARIMA and VAR workflows.
3. Structural breaks around 2021-2022 are visible in several series through the Zivot-Andrews results.
4. No single forecasting model wins for every fuel type.
5. Rolling evaluation generally favors ARIMA for diesel and petrol, while LPG differences are smaller.
6. The second-difference VAR provides a consistent joint-dynamics comparison but has larger errors for diesel and petrol in the current holdout.

These are rolling-origin results, not universal claims about future performance. ARIMAX also assumes future Brent and exchange-rate values are available, so operational use requires forecasts for those external variables.

## 11. How to Run the Analysis

Run the notebooks in this order:

1. `01_data_cleaning_eda.ipynb`
2. `02_stationarity_testing.ipynb`
3. `03_model_identification.ipynb`
4. `04_arima_modeling.ipynb`
5. `05_arimax_modeling.ipynb`
6. `06_forecast_evaluation.ipynb`
7. `07_var_modeling.ipynb`

The modeling notebooks load the already prepared file `../data/dataset_final/full_data.csv`, so notebooks 03-07 can also be run independently as long as the relative working directory is the `notebooks` folder.

The main Python packages used are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## 12. Recommended Next Improvements

For a stronger final research analysis, the next steps are:

1. Implement rolling-origin forecasts at horizons 1-6 months.
2. Add MAPE or sMAPE, while handling values close to zero carefully.
3. Add Diebold-Mariano tests for pairwise forecast-error comparisons.
4. Reconcile the CPI and LPG differencing decisions across the ARIMAX and VAR workflows.
5. Test for cointegration and consider a VECM.
6. Add an explicit 2021-2022 intervention or structural-break specification.
7. Report forecast intervals, not only point forecasts.
8. Compare the model results across multiple rolling windows and consider a VECM if cointegration is supported.

## 13. Diagrams and Visual Outputs

The figures are part of the evidence, not decoration. Each visual answers a different question about the data or the models.

### Exploratory-analysis visuals

- **Time-series plots:** show the long-run movement of each fuel price and external variable. Parallel upward movement suggests common trends, while abrupt changes may indicate shocks or structural breaks.
- **Distribution plots:** show the typical range, skewness, and possible extreme observations for each variable. They help explain why error metrics should be interpreted in the original units of each series.
- **Correlation heatmap:** summarizes linear co-movement in levels. Strong correlations motivate multivariate analysis, but they should not be interpreted as proof of causality because trending series can produce high correlations.
- **Pairwise plots:** show whether two variables move together in a roughly linear way and whether the relationship changes across the sample.

### Stationarity and identification visuals

- **Differenced-series plots:** show whether differencing has removed persistent trend behavior.
- **ACF plots:** display correlation between a series and its lagged values. Slowly declining or significant spikes indicate serial dependence.
- **PACF plots:** display the direct contribution of individual lags after shorter lags are controlled for. Together, ACF and PACF provide starting candidates for ARIMA orders.

### Forecast and model-diagnostic visuals

- **Observed-versus-forecast plots:** compare the fitted model's predictions with the actual holdout observations. A forecast that follows the general direction but misses the turning point still has practical limitations.
- **Residual diagnostic panels:** assess whether residuals resemble unpredictable noise. Remaining autocorrelation suggests that the model has not fully captured the time dependence.
- **Impulse-response diagrams:** in the VAR analysis, trace the estimated response of fuel prices after a one-unit shock to Brent, the exchange rate, or CPI. The confidence bands indicate uncertainty around the estimated response.
- **Forecast-error variance decomposition heatmap:** shows the proportion of a fuel price's forecast uncertainty attributable to each system variable at the selected horizon. Larger values indicate a larger contribution to forecast-error variance, not necessarily a causal effect by themselves.

### How to read the forecasting comparisons

The rolling-origin evaluation reports horizons from one to six months. A one-month horizon predicts the next monthly observation; a six-month horizon predicts the value six months ahead. Errors are calculated repeatedly from different forecast origins so that the comparison is not based on one unusually easy or difficult period.

- **MAE** is the average absolute error and is easy to interpret in GHS/litre or GHS/kg.
- **RMSE** gives more weight to large misses.
- **MAPE** expresses the average error as a percentage, but it can be unstable when actual values are close to zero.
- **Diebold-Mariano tests** compare forecast losses between two models. A small p-value suggests that the observed difference is unlikely to be explained by sampling variation under the test assumptions.

The visuals should therefore be read together with the numerical tables and diagnostics. A model may have a lower error but still show residual problems, while a model such as VAR may be useful for explaining joint dynamics even when it is not the most accurate point forecaster.

## 14. Documentation Map

- `documentation/README.md`: technical record of the completed analysis and interpretation of the notebooks.
- `documentation/CHAPTER_4_README.md`: editable draft of Chapter 4, written in the project's final-year-report style.
- `notebooks/01_data_cleaning_eda.ipynb` to `notebooks/07_var_modeling.ipynb`: main reproducible analysis sequence.
- `data/dataset_final/full_data.csv`: cleaned dataset used by the modeling notebooks.
