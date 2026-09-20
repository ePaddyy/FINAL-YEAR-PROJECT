# CHAPTER FOUR

# RESULTS AND DISCUSSION

## 4.0 Introduction

This chapter presents the results of the empirical analysis of monthly fuel prices in Ghana for the period July 2015 to June 2026. Diesel, petrol, and liquefied petroleum gas (LPG) prices are analysed together with three external variables: the Brent crude oil price, the USD/GHS exchange rate, and the Consumer Price Index (CPI). The analysis follows the sequence set out in Chapter Three: data preparation and exploratory analysis, stationarity testing, model identification, ARIMA estimation, ARIMAX estimation, rolling-origin forecast evaluation, and VAR analysis. Results are reported and interpreted in relation to the study's research objectives, and every reported figure in this chapter is drawn directly from the analysis notebooks (01–07) rather than estimated by hand.

## 4.1 Data Description and Exploratory Results

The cleaned dataset comprises 132 monthly observations spanning July 2015 to June 2026, across six numerical variables: the three fuel-price series and the three external variables. No missing values were present in the final dataset, so no imputation was required.

**Table 4.1: Summary of variable ranges**

| Variable | Minimum | Maximum | Unit |
|---|---:|---:|---|
| Diesel | 2.75 | 17.43 | GHC/litre |
| Petrol | 2.65 | 22.58 | GHS/litre |
| LPG | 3.35 | 15.00 | GHS/kg |
| Brent crude | 26.60 | 117.30 | USD/barrel |
| USD/GHS rate | 3.80 | 15.99 | GHS per USD |
| CPI | 55.00 | 270.80 | Index |

Time-series plots of the six variables show a general upward trajectory over the sample period, punctuated by an especially sharp movement in late 2022, during which diesel and petrol prices briefly reached their sample maxima before reverting. This movement coincides with the depreciation of the Ghanaian cedi and the associated fuel-price crisis documented in Chapter One, and its statistical consequences are examined formally in Section 4.2.

**Figure 4.1: Monthly trends in fuel prices and external variables, July 2015–June 2026**

![Figure 4.1](figures/fig4_1_trends.png)

**Figure 4.2: Distribution of diesel price (bimodal, reflecting the pre/post-2022 regime shift) and Brent crude price (approximately unimodal)**

![Figure 4.2a](figures/fig4_2a_diesel_hist.png)
![Figure 4.2b](figures/fig4_2b_brent_hist.png)

The diesel, petrol, LPG, and exchange-rate distributions are visibly bimodal, with a low pre-2022 cluster and a higher post-2022 cluster — a distributional signature consistent with the structural break formally identified in Section 4.2. Brent crude, by contrast, is closer to unimodal, consistent with its more independent, internationally driven price dynamics.

Level correlations between the fuel-price series are high: diesel and petrol correlate at 0.9909, diesel and LPG at 0.9607, and petrol and LPG at 0.9274. Diesel is also strongly correlated with the exchange rate (0.9551) and CPI (0.9261), while Brent crude correlates more moderately with petrol (0.6883). **Figure 4.3: Correlation heatmap of the level series**

![Figure 4.3](figures/fig4_3_correlation.png)

These high level correlations are consistent with shared long-run trends among the variables rather than evidence of a stable short-run relationship; because trending series can appear strongly correlated even without a genuine structural link (Granger & Newbold, 1974), the level relationships are re-examined using stationary transformations in the sections that follow.

## 4.2 Stationarity Test Results

Three complementary tests were applied to each of the six series: the Augmented Dickey-Fuller (ADF) test (null hypothesis: unit root), the KPSS test (null hypothesis: stationarity), and the Zivot-Andrews (ZA) test, which tests for a unit root while allowing for a single endogenously determined structural break (Zivot & Andrews, 1992).

**Table 4.2: Stationarity test results at level**

| Series | ADF | KPSS | Zivot-Andrews | Estimated break |
|---|---|---|---|---|
| Diesel | Non-stationary | Non-stationary | Stationary with break | 2022-01 |
| LPG | Non-stationary | Non-stationary | Stationary with break | 2022-03 |
| Petrol | Non-stationary | Non-stationary | Stationary with break | 2022-02 |
| Brent | Non-stationary | Non-stationary | Non-stationary | 2021-01 |
| USD/GHS | Non-stationary | Non-stationary | Stationary with break | 2022-03 |
| CPI | Non-stationary | Non-stationary | Non-stationary | 2022-02 |

ADF and KPSS agree that all six series are non-stationary at level. The Zivot-Andrews results add an important qualification: diesel, LPG, petrol, and the exchange rate are each classified as stationary once a single structural break is permitted, with the four estimated break dates clustering tightly around January–March 2022. This is strong, formally tested evidence corroborating the visual and narrative account of the 2022 fuel-price and currency crisis given in Chapter One. Brent crude and CPI remain non-stationary even allowing for a break; this is plausible, since Brent follows international commodity-market dynamics largely independent of the domestic Ghanaian shock, while CPI is a cumulative index that does not revert after a shock by construction.

**Table 4.3: Stationarity after first differencing (d = 1)**

| Series | ADF | KPSS |
|---|---|---|
| Diesel | Stationary | Stationary |
| Petrol | Stationary | Stationary |
| Brent | Stationary | Stationary |
| USD/GHS | Stationary | Stationary |
| LPG | Non-stationary (p = 0.127) | Stationary |
| CPI | Non-stationary | Non-stationary |

Diesel, petrol, Brent, and the exchange rate achieve stationarity after first differencing under both tests. LPG is borderline: KPSS supports stationarity while ADF does not reject the unit-root null at the 5% level (p = 0.127). Given that both tests agreed decisively that LPG was non-stationary at level, and following standard practice of treating KPSS as more informative in this near-unit-root region (Hyndman & Athanasopoulos, 2018), LPG is treated as adequately stationary at d = 1 for the ARIMA and ARIMAX workflows, with this ambiguity noted explicitly here rather than concealed. CPI remains non-stationary at d = 1 under both tests and requires second differencing (ADF p = 0.0087, KPSS p = 0.10 at d = 2).

Because the VAR model requires a common order of integration across all system variables, and CPI is part of that system, all six variables entering the VAR model — diesel, petrol, LPG, Brent, the exchange rate, and CPI — are second-differenced for consistency, even though most of them were already stationary at d = 1 individually. This is a standard and conservative choice: a series stationary at d = 1 remains stationary at d = 2, at the modest cost of slightly noisier differenced values. ARIMA and ARIMAX retain each series' own correct individual order (d = 1 in all three cases).

The clustering of Zivot-Andrews break dates around early 2022 is treated as a substantive finding rather than a nuisance: it is referenced again in Sections 4.3 and 4.6 to explain patterns observed in the residual diagnostics of the fitted models.

## 4.3 ARIMA Model Results

**Figure 4.4: ACF and PACF of the first-differenced diesel price series** (petrol and LPG show a similar low-order cutoff pattern; see Notebook 03)

![Figure 4.4](figures/fig4_4_acf_pacf_diesel.png)

Univariate ARIMA models were fitted to each fuel-price series using its own history only. The final six months of the sample (January–June 2026) were withheld as an out-of-sample test set, with the preceding 126 months (July 2015–December 2025) used for training and order selection. Candidate orders (p, 1, q) with p, q ∈ {0, 1, 2, 3} were compared using training-set AIC.

**Table 4.4: ARIMA model results (six-month holdout)**

| Series | Selected order | MAE | RMSE | MAPE | Ljung-Box p (lag 10) |
|---|---|---:|---:|---:|---:|
| Diesel | (0, 1, 3) | 0.8652 | 1.0243 | 6.17% | 0.9699 |
| Petrol | (0, 1, 3) | 1.4638 | 1.6534 | 9.38% | 0.9934 |
| LPG | (2, 1, 2) | 0.1888 | 0.2293 | 1.40% | 0.0338 |

**Figure 4.5: ARIMA(0,1,3) forecast against observed values, diesel price (six-month holdout)**

![Figure 4.5](figures/fig4_5_arima_diesel_forecast.png)

Diesel and petrol residuals show no evidence of remaining autocorrelation (Ljung-Box p > 0.9 in both cases), indicating that the fitted ARIMA structures adequately capture the short-run dependence in these series. The LPG model is the exception: its Ljung-Box p-value (0.034) falls below the 5% threshold, indicating that some serial dependence remains unexplained by ARIMA(2,1,2). This is addressed directly in Section 4.4, where the addition of external regressors is shown to resolve the issue.

Petrol has the largest absolute errors of the three series, though this partly reflects its wider price range and the extreme 2022 spike (petrol's series maximum of GHS 22.58/litre is markedly above its typical range). LPG produces the smallest errors in absolute terms.

## 4.4 ARIMAX Model Results

The ARIMAX specification extends each ARIMA model with Brent crude oil prices and the USD/GHS exchange rate as exogenous regressors, motivated by the theoretical channel through which international oil prices and currency depreciation feed into domestic pump prices. The same training/test split and order-search procedure as Section 4.3 were used.

**Table 4.5: ARIMAX model results (six-month holdout)**

| Series | Selected order | MAE | RMSE | MAPE | Ljung-Box p (lag 10) |
|---|---|---:|---:|---:|---:|
| Diesel | (0, 1, 3) | 0.4769 | 0.5526 | 3.49% | 0.7909 |
| Petrol | (0, 1, 3) | 0.7162 | 0.7801 | 4.65% | 0.9618 |
| LPG | (1, 1, 3) | 0.1136 | 0.1349 | 0.84% | 0.4731 |

**Figure 4.6: ARIMAX(0,1,3) forecast against observed values, diesel price (six-month holdout)**

![Figure 4.6](figures/fig4_6_arimax_diesel_forecast.png)

ARIMAX reduces both MAE and RMSE relative to ARIMA for all three fuel series in the six-month holdout, with the largest proportional improvement for diesel and petrol (MAE roughly halved in both cases). The residual autocorrelation problem identified in the LPG ARIMA model is resolved: the LPG ARIMAX Ljung-Box p-value rises to 0.473, indicating no remaining serial dependence once Brent and the exchange rate are included. This result supports the study's third objective — that external, economically motivated predictors carry information beyond the fuel price's own history.

This result must be interpreted conditionally. The reported forecasts supply the model with the actual observed Brent and exchange-rate values for the test period; in an operational forecasting setting, these external variables would themselves need to be forecast or otherwise estimated, and any uncertainty in those inputs would propagate into the fuel-price forecast. The six-month holdout result therefore demonstrates the *information content* of Brent and the exchange rate for fuel-price forecasting, rather than the accuracy achievable in a fully unconditional, real-time forecasting exercise.

## 4.5 Rolling-Origin Forecast Evaluation

To assess forecast accuracy more robustly than a single six-month holdout allows, a rolling-origin evaluation was conducted. Starting from an initial training window of 96 months, ARIMA, ARIMAX, and VAR orders were identified once from this initial window and then re-fitted (using the fixed order) at each of 31 successive forecast origins, producing forecasts at horizons of one to six months from each origin. This design follows standard rolling-origin evaluation practice (Hyndman & Athanasopoulos, 2018) while keeping model orders fixed across origins for computational tractability — a simplification made explicit here rather than left implicit.

Errors were summarised using MAE, RMSE, and MAPE at each horizon, and Diebold-Mariano (Diebold & Mariano, 1995) tests were used to assess whether differences in squared-error loss between model pairs were statistically distinguishable from sampling variation, using a HAC-adjusted variance with a bandwidth equal to (horizon − 1) lags, following the original Diebold-Mariano recommendation for h-step-ahead forecast errors.

**Table 4.6: Cumulative rolling-origin accuracy at selected horizons (diesel)**

| Model | Horizon (months) | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| ARIMA | 1 | 0.3868 | 0.5469 | 3.01% |
| ARIMA | 6 (cumulative) | 0.7422 | 0.9618 | 5.62% |
| ARIMAX | 1 | 0.5235 | 0.6208 | 4.01% |
| ARIMAX | 6 (cumulative) | 1.0302 | 1.1658 | 7.76% |
| VAR | 1 | 0.9300 | 1.2751 | 7.11% |
| VAR | 6 (cumulative) | 1.7789 | 2.3032 | 13.56% |

**Figure 4.7: Rolling-origin MAE by forecast horizon, 31 forecast origins**

![Figure 4.7](figures/fig4_7_rolling_origin_mae.png)

Figure 4.7 makes the rolling-origin pattern visually clear: ARIMA's error curve sits consistently below both ARIMAX and VAR at every horizon for diesel and petrol, with the gap widening as the horizon extends to six months. For LPG, ARIMA and ARIMAX track closely at short horizons before diverging after horizon 3, while VAR's LPG error grows fastest of the three models beyond horizon 4.

A representative Diebold-Mariano comparison, at the one-month cumulative horizon for diesel, found ARIMA significantly outperformed VAR (DM = −2.25, p = 0.024) and outperformed ARIMAX at a level close to but not reaching significance (DM = −0.93, p = 0.353), while for petrol, ARIMA significantly outperformed both ARIMAX (DM = −2.50, p = 0.012) and VAR (DM = −2.70, p = 0.007) at the one-month horizon. The full set of pairwise comparisons across all horizons and series is reported in Notebook 06.

This result appears, at first glance, to contradict the six-month single-holdout finding in Section 4.4, where ARIMAX outperformed ARIMA. The two results are not actually in conflict: they answer different questions. The single holdout in Sections 4.3–4.4 reflects performance in one specific final period (January–June 2026); the rolling-origin evaluation summarises performance repeatedly, averaged across 31 different forecast origins spanning a much longer and more varied period. The rolling result indicates that ARIMAX's advantage over ARIMA in the final six-month window is not consistently replicated across the full evaluation history, whereas the single-holdout result shows that, specifically for the most recent period, adding Brent and the exchange rate did help. Both findings are retained and reported together, as each carries different practical implications: the rolling result speaks to average historical reliability, while the holdout result speaks to recent, current-regime performance.

## 4.6 VAR Model Results

The VAR model treats the three fuel prices, Brent crude, the exchange rate, and CPI as a single interdependent six-variable system. Because CPI remains non-stationary after first differencing (Section 4.2), the complete system was estimated in second differences, and forecasts were integrated twice to return to the original price levels. Candidate lag lengths from one to six months were evaluated, restricted to lags producing a stable VAR system, with the lowest-AIC stable candidate selected.

**Table 4.7: VAR lag-order selection**

| Lag | AIC | BIC |
|---:|---:|---:|
| 2 | −4.857 | −3.026 (BIC-preferred) |
| 6 | −5.800 (AIC-preferred) | −0.588 |

AIC and BIC disagree: AIC favours the higher-order lag-6 specification, while BIC, which penalises additional parameters more heavily, favours the more parsimonious lag 2. Given the modest effective sample size (approximately 118–122 usable observations for a six-variable system after second differencing), this disagreement raises a legitimate concern about overparameterization at lag 6, where each of the six equations estimates 37 parameters (an observations-to-parameters ratio of roughly 3.2), compared with 13 parameters per equation at lag 2 (a ratio of roughly 9.4).

To resolve this, both specifications were fitted and compared directly on residual diagnostics and six-month holdout forecast accuracy.

**Table 4.8: Lag-2 versus lag-6 robustness comparison**

| Lag | Obs./parameter | Whiteness stat | Normality stat | Diesel MAE | Petrol MAE | LPG MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9.38 | 473.06 (p < 0.001) | 861.07 (p < 0.001) | 1.9805 | 2.8416 | 0.5031 |
| 6 | 3.19 | 277.51 (p < 0.001) | 251.34 (p < 0.001) | 2.0456 | 2.2219 | 0.2198 |

Contrary to the initial overparameterization concern, reducing the lag order to 2 did not improve residual behaviour — normality was in fact substantially *more* strongly rejected at lag 2 (statistic 861 versus 251) — and produced materially weaker six-month forecast accuracy for petrol and LPG. This pattern suggests that the residual non-normality present in the VAR system at both lag lengths is driven primarily by the 2021–2022 structural shock identified via the Zivot-Andrews test (Section 4.2), which a lower-order VAR captures even less completely than a higher-order one, rather than being a symptom of lag-6 overparameterization. On the combined weight of AIC-optimality, superior forecast accuracy, and the finding that parsimony does not resolve the residual issue, lag 6 is retained as the final VAR specification.

**Table 4.9: VAR forecast results (lag 6, six-month holdout)**

| Series | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Diesel | 2.0456 | 2.1683 | 14.77% |
| Petrol | 2.2219 | 2.3378 | 14.39% |
| LPG | 0.2198 | 0.2757 | 1.63% |

VAR produces larger errors than both ARIMA and ARIMAX for diesel and petrol in the six-month holdout, and is not the preferred point-forecasting model for these series. Its value lies elsewhere: in its capacity to characterise joint dynamics among the system variables, which ARIMA and ARIMAX, as univariate and conditionally univariate models respectively, cannot address.

**Residual diagnostics.** The multivariate residual whiteness test rejects the null of no remaining autocorrelation (test statistic 277.51 against a critical value of 173.00, p < 0.001), and the multivariate normality test likewise rejects (test statistic 251.34 against a critical value of 21.03, p < 0.001). As established above, these violations persist even at the more parsimonious lag 2, supporting the interpretation that they reflect the 2021–2022 structural shock rather than a correctable specification error. The VAR system should accordingly be read as an informative benchmark for system-level dynamics rather than as a residually adequate forecasting model in its own right.

**Granger causality.** Pairwise Granger-causality tests (Table 4.10) examine whether the lagged values of each system variable help predict each fuel-price series, conditional on the fitted lag-6 VAR.

**Table 4.10: Pairwise Granger-causality results (causing → target)**

| Causing variable | Target variable | F-statistic | p-value | Significant (5%) |
|---|---|---:|---:|:---:|
| USD/GHS rate | Diesel | 6.98 | < 0.001 | Yes |
| LPG | Diesel | 3.96 | 0.001 | Yes |
| Petrol | Diesel | 2.29 | 0.034 | Yes |
| Brent crude | Diesel | 2.22 | 0.040 | Yes |
| CPI | Diesel | 0.79 | 0.574 | No |
| Petrol | LPG | 5.73 | < 0.001 | Yes |
| USD/GHS rate | LPG | 4.14 | < 0.001 | Yes |
| Diesel | LPG | 3.13 | 0.005 | Yes |
| CPI | LPG | 1.24 | 0.286 | No |
| Brent crude | LPG | 0.52 | 0.792 | No |
| USD/GHS rate | Petrol | 6.88 | < 0.001 | Yes |
| LPG | Petrol | 2.95 | 0.008 | Yes |
| Diesel | Petrol | 1.81 | 0.096 | No |
| Brent crude | Petrol | 1.32 | 0.247 | No |
| CPI | Petrol | 0.44 | 0.851 | No |

The exchange rate Granger-causes all three fuel prices at the 5% level, and each fuel price Granger-causes the others, consistent with the strong level correlations reported in Section 4.1 and with a common exchange-rate transmission channel. Brent crude Granger-causes diesel but not LPG or petrol, and CPI does not Granger-cause any of the three fuel prices at the 5% level. These results should be read as statements about predictive content within the fitted VAR system, not as claims of structural causal mechanisms (Granger, 1969).

**Impulse responses and variance decomposition.** Impulse-response functions (12-step horizon, Monte Carlo error bands, 500 replications) and forecast-error variance decomposition (6-step horizon) were also produced (Notebook 07) to characterise the estimated dynamic response of each fuel price to shocks in Brent, the exchange rate, and CPI, and the proportion of each fuel price's six-month forecast-error variance attributable to each system variable.

**Figure 4.8: Impulse responses of fuel prices to Brent, exchange-rate, and CPI shocks**

![Figure 4.8](figures/fig4_8_var_irf.png)

The impulse-response diagrams show diesel and petrol responding positively and with visible oscillation to both Brent and exchange-rate shocks, with the exchange-rate response noticeably larger in magnitude than the Brent response for both fuels — consistent with the stronger Granger-causality result for the exchange rate reported in Table 4.10. LPG's responses to all three shocks are smaller in magnitude and less clearly distinguishable from zero, given its wider confidence bands.

**Figure 4.9: Six-month forecast-error variance decomposition**

![Figure 4.9](figures/fig4_9_var_fevd.png)

The variance decomposition produces a notable result: **CPI accounts for the largest share of six-month forecast-error variance for all three fuel prices** — 68% for diesel, 68% for petrol, and 65% for LPG — substantially exceeding each fuel's own shock (23%, 18%, and 16% respectively) and far exceeding the direct contributions of Brent crude and the exchange rate (each 8% or less across all three fuels). This is a striking finding given that CPI did not Granger-cause any of the three fuel prices in Table 4.10's pairwise tests. The two results are not contradictory: Granger causality in Table 4.10 tests only the *direct*, one-to-one predictive relationship between CPI and each fuel price, whereas the FEVD reflects CPI's *total* contribution to forecast-error variance through the full dynamic system, including indirect transmission through the exchange rate and the other fuel prices, to which CPI is itself strongly correlated (Section 4.1) and dynamically linked at lag 6. This suggests CPI's role in the system is best understood as a broad indicator of the same inflationary and currency pressures driving fuel prices, transmitted indirectly, rather than as a direct short-run predictor in its own right. This finding is presented as a descriptive, model-conditional output rather than independent causal evidence, consistent with the caveats already noted for the Granger-causality results above.

## 4.7 Discussion of Findings by Objective

**Objective 1: Describe the statistical properties and trends.** The exploratory analysis (Section 4.1) documents strong common upward movement across all six variables, with an unusually sharp episode in late 2022. The stationarity analysis (Section 4.2) confirms all six series are non-stationary in levels, and the Zivot-Andrews results formally corroborate that this episode constitutes a structural break for diesel, LPG, petrol, and the exchange rate.

**Objective 2: Develop ARIMA, ARIMAX, and VAR models.** All three model families were implemented and diagnosed. Differencing orders were established through the stationarity testing sequence, autoregressive and moving-average orders through ACF/PACF inspection and AIC comparison, and the VAR lag order through a combination of AIC/BIC comparison, stability checking, and the direct lag-2/lag-6 robustness comparison in Section 4.6.

**Objective 3: Determine whether external shocks improve forecasts.** The six-month holdout comparison (Sections 4.3–4.4) shows ARIMAX improving on ARIMA for all three fuel series, most substantially for diesel and petrol, and resolving the residual autocorrelation present in the LPG ARIMA model. This finding is qualified by the rolling-origin result (Section 4.5), which shows ARIMA outperforming ARIMAX across a longer, repeated evaluation history for diesel and petrol — indicating that the value of external regressors is real but not uniformly present across all forecasting conditions.

**Objective 4: Compare accuracy across one- to six-month horizons.** The rolling-origin procedure (Section 4.5) provides horizon-specific error metrics and Diebold-Mariano comparisons. Across the sampled horizons, ARIMA is the more consistently accurate univariate approach for diesel and petrol; LPG shows smaller, less decisive differences between models.

**Objective 5: Assess practical forecasting usefulness.** All three model families can provide useful advance information about likely fuel-price movement, particularly at short horizons, but none should be treated as delivering certain point forecasts. ARIMAX's six-month result is conditional on external-variable values that would themselves require forecasting in practice; the VAR system's residual diagnostics indicate it is better suited to system-level analysis than to standalone point forecasting; and forecast uncertainty generally increases with horizon across all three approaches.

## 4.8 Summary of the Main Results

1. The six fuel-price and macroeconomic series exhibit strong upward movement and high level correlation over the sample period, with an unusually sharp episode in late 2022.
2. All six series are non-stationary in levels; Zivot-Andrews testing formally identifies a structural break clustering around January–March 2022 for diesel, LPG, petrol, and the exchange rate.
3. ARIMA provides a credible univariate baseline; diesel and petrol residuals show no remaining autocorrelation, while LPG residuals retain some serial dependence (Ljung-Box p = 0.034).
4. ARIMAX reduces six-month holdout forecast errors relative to ARIMA for all three fuel products and resolves the LPG residual autocorrelation issue.
5. Rolling-origin evaluation across 31 forecast origins presents a more mixed picture, generally favouring ARIMA over ARIMAX for diesel and petrol.
6. A direct lag-2 versus lag-6 robustness comparison shows that VAR's residual non-normality is not resolved, and is in fact worsened, by reducing model complexity — supporting the interpretation that it reflects the 2021–2022 structural shock rather than overparameterization, and justifying the retention of the AIC-optimal lag-6 specification.
7. VAR's six-month point forecasts are less accurate than ARIMA or ARIMAX for diesel and petrol, but the model provides Granger-causality, impulse-response, and variance-decomposition evidence unavailable from the univariate approaches — most notably, that the exchange rate Granger-causes all three fuel prices.
8. Forecast-error variance decomposition shows CPI contributing the largest share (65–68%) of six-month forecast-error variance for all three fuel prices — well above each fuel's own shock or the direct contributions of Brent and the exchange rate — despite not directly Granger-causing any fuel price, indicating an indirect transmission channel through the system's other variables rather than a direct predictive relationship.

## 4.9 Chapter Conclusion

This chapter has presented and interpreted the empirical results of the Ghana fuel-price forecasting study. Fuel prices over the study period are characterised by persistent trend behaviour, strong common movement with macroeconomic conditions, and a clearly identifiable structural disruption around 2021–2022. No single model dominates across every product and every forecast horizon: ARIMAX offers the strongest single-period result when Brent and exchange-rate information are available, ARIMA is the more consistently reliable univariate approach across repeated rolling evaluation, and VAR, while not the most accurate point forecaster, provides system-level insight — particularly the finding that the exchange rate Granger-causes all three fuel prices — unavailable from the other two approaches. The appropriate model therefore depends on whether the priority is univariate robustness, conditional forecasting accuracy, or the analysis of joint macroeconomic dynamics.

Chapter Five draws the final conclusions of the study, states its limitations, and provides recommendations for policymakers, consumers, analysts, and future researchers.
