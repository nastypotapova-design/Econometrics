import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 10
# ----------------------------------------------------------------------------------------------------------------------
# Stage 1.Analysis of Price Series.
# ----------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('data_part1.csv')

df = df.drop(columns=['Unnamed: 0'])
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df.sort_index(inplace=True)

train = df.iloc[:-20]
test = df.iloc[-20:]

if not os.path.exists('figures'):
    os.makedirs('figures')

price = train['portfolio']

plt.figure(figsize=(12, 6))
plt.plot(price, linewidth=1.5, color='darkblue')
plt.title('Portfolio Price Series (Training Sample)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/price_series.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate ACF and PACF values for first 5 lags
acf_values = acf(price, nlags=5)
pacf_values = pacf(price, nlags=5)


# ACF plot
fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(price, lags=40, ax=ax, alpha=0.05)
ax.set_title('Autocorrelation Function (ACF) - Portfolio Price Series', fontsize=14)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)

ax.set_ylim(0, 1)

ax.axhline(y=1.96/np.sqrt(len(price)), color='blue', linestyle='--', alpha=0.5)
ax.text(35, 1.96/np.sqrt(len(price)) + 0.02, '95% confidence band', fontsize=9, alpha=0.7)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/price_acf.png', dpi=300, bbox_inches='tight')
plt.close()


# PACF plot
fig, ax = plt.subplots(figsize=(12, 6))
plot_pacf(price, lags=40, ax=ax, alpha=0.05, method='ywm')
ax.set_title('Partial Autocorrelation Function (PACF) - Portfolio Price Series', fontsize=14)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Partial Autocorrelation', fontsize=12)

ax.set_ylim(-0.3, 1)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/price_pacf.png', dpi=300, bbox_inches='tight')
plt.close()
# Dickey-Fuller Test for Price
adf_result = adfuller(price.dropna())

# -----------------------------------------------------------------------------------------------------------------------
# Stage 2. Returns Series Calculation and Analysis
# -----------------------------------------------------------------------------------------------------------------------
# Calculate returns
returns = price.pct_change().dropna()

# Plot returns series
plt.figure(figsize=(12, 6))
plt.plot(returns, linewidth=1.2, color='green')
plt.title('Portfolio Returns Series (Training Sample)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/returns_series.png', dpi=300, bbox_inches='tight')
plt.close()

# ACF and PACF for Returns

acf_returns = acf(returns, nlags=5)
pacf_returns = pacf(returns, nlags=5)


# ACF plot for returns
fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(returns, lags=20, ax=ax, alpha=0.05)
ax.set_title('Autocorrelation Function (ACF) - Portfolio Returns', fontsize=14)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)

conf_band_returns = 1.96 / np.sqrt(len(returns))
ax.axhline(y=conf_band_returns, color='blue', linestyle='--', alpha=0.5)
ax.axhline(y=-conf_band_returns, color='blue', linestyle='--', alpha=0.5)
ax.text(18, conf_band_returns + 0.02, '95% confidence band', fontsize=9, alpha=0.7)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/returns_acf.png', dpi=300, bbox_inches='tight')
plt.close()

# PACF plot for returns
fig, ax = plt.subplots(figsize=(12, 6))
plot_pacf(returns, lags=20, ax=ax, alpha=0.05, method='ywm')
ax.set_title('Partial Autocorrelation Function (PACF) - Portfolio Returns', fontsize=14)
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Partial Autocorrelation', fontsize=12)

# Add confidence band annotation
ax.axhline(y=conf_band_returns, color='blue', linestyle='--', alpha=0.5)
ax.axhline(y=-conf_band_returns, color='blue', linestyle='--', alpha=0.5)
ax.text(18, conf_band_returns + 0.02, '95% confidence band', fontsize=9, alpha=0.7)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/returns_pacf.png', dpi=300, bbox_inches='tight')
plt.close()

# Dickey-Fuller Test for Returns
adf_returns = adfuller(returns.dropna())
# ----------------------------------------------------------------------------------------------------------------------
# Stage 3. ARIMA Modeling
# ----------------------------------------------------------------------------------------------------------------------

# Test only relevant models based on ACF/PACF patterns
models_to_test = [(0,0,0), (1,0,0), (0,0,1), (1,0,1)]
results = []

for p, d, q in models_to_test:
    model = ARIMA(returns, order=(p, d, q))
    fitted = model.fit()
    results.append({
        'order': (p, d, q),
        'aic': fitted.aic,
        'model': fitted
    })

# Find best model
best_result = min(results, key=lambda x: x['aic'])
best_order = best_result['order']
best_fitted = best_result['model']


# Residual Diagnostics

residuals = best_fitted.resid

# ACF of residuals
fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(residuals, lags=20, ax=ax, alpha=0.05)
ax.set_title('ACF of Residuals', fontsize=14)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/residuals_acf.png', dpi=300, bbox_inches='tight')
plt.close()

# Residual density plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_title('Residuals Distribution', fontsize=14)
ax.set_xlabel('Residuals')
ax.set_ylabel('Density')
ax.grid(True, alpha=0.3)


x = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x, norm.pdf(x, residuals.mean(), residuals.std()), 'r-', linewidth=2, label='Normal')
ax.legend()
plt.tight_layout()
plt.savefig('figures/residuals_density.png', dpi=300, bbox_inches='tight')
plt.close()


# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[5, 10, 15], return_df=True)
# ----------------------------------------------------------------------------------------------------------------------
# Stage 4. Forecasting (20-step ahead)
# ----------------------------------------------------------------------------------------------------------------------

# Use the best model (ARIMA(1,0,0)) for forecasting
forecast_steps = 20
forecast = best_fitted.forecast(steps=forecast_steps)

# Get actual returns for the test period
test_returns = test['portfolio'].pct_change().dropna()

actual_forecast = test_returns.iloc[:forecast_steps]

forecast_aligned = forecast
actual_aligned = actual_forecast

comparison = pd.DataFrame({
    'Forecast': forecast_aligned.values[:10],
    'Actual': actual_aligned.values[:10]
}, index=actual_aligned.index[:10])


# Plot forecast vs actual
plt.figure(figsize=(12, 6))

# Plot historical returns (last 50 days for context)
plt.plot(returns.index[-50:], returns.iloc[-50:],
         color='gray', linewidth=1.5, label='Historical Returns (last 50 days)')

# Plot forecast (aligned) — используем даты из actual_aligned
plt.plot(actual_aligned.index, forecast_aligned, color='blue', linewidth=2,
         marker='o', markersize=4, label='Forecast')

# Plot actual test returns (aligned)
plt.plot(actual_aligned.index, actual_aligned, color='red', linewidth=2,
         marker='s', markersize=4, label='Actual Returns')

plt.title('20-Step Ahead Forecast vs Actual Returns', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Returns', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/forecast_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate forecast accuracy metrics
mae = mean_absolute_error(actual_aligned, forecast_aligned)
rmse = np.sqrt(mean_squared_error(actual_aligned, forecast_aligned))

# Compare with naive forecast (zero)
naive_mae = mean_absolute_error(actual_aligned, [0] * len(actual_aligned))
naive_rmse = np.sqrt(mean_squared_error(actual_aligned, [0] * len(actual_aligned)))

# ----------------------------------------------------------------------------------------------------------------------
# Record all results in README
# ----------------------------------------------------------------------------------------------------------------------

with open('README.md', 'w', encoding='utf-8') as f:
    f.write('# Econometrics Project: Portfolio Return Modeling\n\n')

    f.write('This document presents **Part 1: Portfolio Return Modeling**. ')
    f.write('The analysis includes price series examination, returns calculation, ARIMA modeling, and forecasting.\n\n')
    f.write('## Data Overview\n\n')
    f.write(f'- **Total observations:** {len(df)}\n')
    f.write(f'- **Training sample:** {len(train)} observations (first {len(train)})\n')
    f.write(f'- **Test sample:** {len(test)} observations (last 20, reserved for forecasting)\n')
    f.write(f'- **Time period:** {df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")}\n\n')
    f.write('**Portfolio composition:**\n\n')
    f.write('- 25% Apple (AAPL)\n')
    f.write('- 15% Tesla (TSLA)\n')
    f.write('- 20% Yandex (YNDX)\n')
    f.write('- 20% Google (GOOGL)\n')
    f.write('- 20% Boeing (BA)\n\n')

    f.write('---\n\n')
    # ------------------------------------------------------------------------------------------------------------------
    # Stage 1. Analysis of Price Series
    # ------------------------------------------------------------------------------------------------------------------
    f.write('### 1.1 Time Series Plot\n\n')
    f.write('![Price Series](figures/price_series.png)\n\n')
    f.write('**Observations:**\n\n')
    f.write(f'- **Trend:** The portfolio price shows a strong upward trend from ${price.min():.2f} ')
    f.write(f'(April 2020) to ${price.max():.2f} (February 2021), more than doubling in value.\n')
    f.write('- **Seasonality:** No clear seasonal pattern is visible at the daily frequency.\n')
    f.write('- **Volatility:** The series shows periods of rapid growth followed by declines.\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Stage 1.2 ACF and PACF
    # ------------------------------------------------------------------------------------------------------------------
    f.write('### 1.2 Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)\n\n')

    f.write('**ACF values (first 5 lags):**\n\n')
    f.write('| Lag | ACF |\n')
    f.write('|-----|-----|\n')
    for i in range(1, 6):
        f.write(f'| {i} | {acf_values[i]:.4f} |\n')

    f.write('\n**PACF values (first 5 lags):**\n\n')
    f.write('| Lag | PACF |\n')
    f.write('|-----|------|\n')
    for i in range(1, 6):
        f.write(f'| {i} | {pacf_values[i]:.4f} |\n')

    f.write('\n**ACF Plot:**\n\n')
    f.write('![ACF Price Series](figures/price_acf.png)\n\n')
    f.write('**PACF Plot:**\n\n')
    f.write('![PACF Price Series](figures/price_pacf.png)\n\n')


    f.write('**Interpretation:**\n\n')
    f.write(f'- The ACF values remain high across all five lags (from {acf_values[1]:.4f} to {acf_values[5]:.4f}), ')
    f.write(
        'meaning that past prices are strongly correlated with current prices even after several days.\n')
    f.write(f'- The PACF shows a strong value of {pacf_values[1]:.4f} at lag 1, followed by values close to zero ')
    f.write(
        f'({pacf_values[2]:.4f}, {pacf_values[3]:.4f}, {pacf_values[4]:.4f}, {pacf_values[5]:.4f}) at lags 2 through 5. ')
    f.write(
        "This means that after accounting for yesterday's price, earlier days have almost no additional influence.\n\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Stage 1.3 Dickey-Fuller Test
    # ------------------------------------------------------------------------------------------------------------------
    f.write('### 1.3 Augmented Dickey-Fuller (ADF) Test\n\n')
    f.write('| Statistic | Value |\n')
    f.write('|-----------|-------|\n')
    f.write(f'| ADF Statistic | {adf_result[0]:.4f} |\n')
    f.write(f'| p-value | {adf_result[1]:.4f} |\n')
    f.write(f'| Critical Value (1%) | {adf_result[4]["1%"]:.4f} |\n')
    f.write(f'| Critical Value (5%) | {adf_result[4]["5%"]:.4f} |\n')
    f.write(f'| Critical Value (10%) | {adf_result[4]["10%"]:.4f} |\n')
    f.write(f'| Number of lags used | {adf_result[2]} |\n')
    f.write(f'| Number of observations | {adf_result[3]} |\n\n')

    f.write('**Conclusion:**\n\n')
    f.write(f'The ADF test statistic ({adf_result[0]:.4f}) is greater than all critical values, ')
    f.write(f'and the p-value ({adf_result[1]:.4f}) is well above the 0.05 significance level. ')
    f.write('Therefore, we **fail to reject the null hypothesis** of a unit root. ')
    f.write('This confirms that the price series is **non-stationary**, meaning it does not have a constant mean or variance over time.\n\n')

    f.write('---\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Stage 2. Analysis of Returns Series
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Stage 2. Analysis of Returns Series\n\n')

    # 2.1 Returns Time Series Plot
    f.write('### 2.1 Returns Time Series Plot\n\n')
    f.write('![Returns Series](figures/returns_series.png)\n\n')
    f.write('**Observations:**\n\n')
    f.write('- Unlike the price series, returns fluctuate around zero and do not show a clear upward or downward trend.\n')
    f.write('- Periods of large movements are followed by periods of smaller movements.\n')
    f.write(f'- The returns range from about {returns.min()*100:.1f}% to {returns.max()*100:.1f}%.\n')

    # 2.2 Summary Statistics
    f.write('### 2.2 Summary Statistics\n\n')
    f.write('| Statistic | Value |\n')
    f.write('|-----------|-------|\n')
    f.write(f'| Mean Return | {returns.mean():.6f} ({returns.mean() * 100:.4f}%) |\n')
    f.write(f'| Standard Deviation | {returns.std():.6f} ({returns.std() * 100:.4f}%) |\n')
    f.write(f'| Minimum | {returns.min():.6f} ({returns.min() * 100:.4f}%) |\n')
    f.write(f'| Maximum | {returns.max():.6f} ({returns.max() * 100:.4f}%) |\n\n')

    f.write(
        f'The average daily return of {returns.mean() * 100:.4f}% reflects the strong upward trend observed in the price series.\n\n')

    # 2.3 ACF and PACF for Returns
    f.write('### 2.3 Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for Returns\n\n')

    f.write('**ACF values (first 5 lags):**\n\n')
    f.write('| Lag | ACF |\n')
    f.write('|-----|-----|\n')
    for i in range(1, 6):
        f.write(f'| {i} | {acf_returns[i]:.4f} |\n')

    f.write('\n**PACF values (first 5 lags):**\n\n')
    f.write('| Lag | PACF |\n')
    f.write('|-----|------|\n')
    for i in range(1, 6):
        f.write(f'| {i} | {pacf_returns[i]:.4f} |\n')

    conf_band_returns_val = 1.96 / np.sqrt(len(returns))
    f.write(
        f'\n**Note:** With {len(returns)} observations, the 95% confidence band is approximately **±{conf_band_returns_val:.3f}**. ')
    f.write('Values within this range are not statistically significant.\n\n')

    f.write('**ACF Plot:**\n\n')
    f.write('![ACF Returns](figures/returns_acf.png)\n\n')
    f.write('**PACF Plot:**\n\n')
    f.write('![PACF Returns](figures/returns_pacf.png)\n\n')

    f.write('**Interpretation:**\n\n')
    f.write(
        f'- The ACF at lag 1 is {acf_returns[1]:.4f}, which is within the 95% confidence band (±{conf_band_returns_val:.3f}). ')
    f.write(
        'This means the correlation between today\'s return and yesterday\'s return is not statistically significant.\n')
    f.write(f'- All other lags also have values close to zero and remain within the confidence band.\n')
    f.write(
        '- The PACF shows a similar pattern: a small negative value at lag 1 and no significant spikes afterward.\n')
    f.write(
        '- This pattern indicates that returns have **no significant autocorrelation**, meaning past returns do not help predict future returns.\n\n')

    # 2.4 Dickey-Fuller Test for Returns
    f.write('### 2.4 Augmented Dickey-Fuller (ADF) Test for Returns\n\n')
    f.write('| Statistic | Value |\n')
    f.write('|-----------|-------|\n')
    f.write(f'| ADF Statistic | {adf_returns[0]:.4f} |\n')
    f.write(f'| p-value | {adf_returns[1]:.4f} |\n')
    f.write(f'| Critical Value (1%) | {adf_returns[4]["1%"]:.4f} |\n')
    f.write(f'| Critical Value (5%) | {adf_returns[4]["5%"]:.4f} |\n')
    f.write(f'| Critical Value (10%) | {adf_returns[4]["10%"]:.4f} |\n')
    f.write(f'| Number of lags used | {adf_returns[2]} |\n')
    f.write(f'| Number of observations | {adf_returns[3]} |\n\n')

    f.write('**Conclusion:**\n\n')
    f.write(f'The ADF test statistic ({adf_returns[0]:.4f}) is far less than the critical value at the 1% level, ')
    f.write(f'and the p-value ({adf_returns[1]:.4f}) is effectively zero. ')
    f.write('Therefore, we **reject the null hypothesis** of a unit root. ')
    f.write('The returns series is **stationary**, which satisfies a key requirement for ARIMA modeling.\n\n')

    f.write('---\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Stage 3. ARIMA Modeling
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Stage 3. ARIMA Modeling\n\n')

    # 3.1 Model Selection
    f.write('### 3.1 Model Selection\n\n')
    f.write('Based on the ACF/PACF patterns of returns, the following models were estimated:\n\n')
    f.write('| Model | AIC |\n')
    f.write('|-------|-----|\n')
    for res in sorted(results, key=lambda x: x['aic']):
        p, d, q = res['order']
        f.write(f'| ARIMA({p},{d},{q}) | {res["aic"]:.2f} |\n')

    best_p, best_d, best_q = best_order
    f.write(f'\n**Selected model:** ARIMA({best_p},{best_d},{best_q}) with AIC = {best_result["aic"]:.2f}\n\n')
    f.write('The AR(1) specification provides the best fit according to the Akaike Information Criterion.\n\n')

    # 3.2 Model Coefficients
    f.write('### 3.2 Model Coefficients\n\n')
    f.write('| Coefficient | Estimate | Std Error | z-statistic | p-value |\n')
    f.write('|-------------|----------|-----------|-------------|---------|\n')

    for name in best_fitted.params.index:
        z_stat = best_fitted.params[name] / best_fitted.bse[name]
        p_val = best_fitted.pvalues[name]
        sig = ''
        if p_val < 0.01:
            sig = '***'
        elif p_val < 0.05:
            sig = '**'
        elif p_val < 0.1:
            sig = '*'
        f.write(
            f'| {name} | {best_fitted.params[name]:.6f} | {best_fitted.bse[name]:.6f} | {z_stat:.4f} | {p_val:.4f}{sig} |\n')

    f.write('\n*Significance codes: *** p<0.01, ** p<0.05, * p<0.1*\n\n')

    f.write('**Interpretation:**\n\n')
    f.write(f'- The constant term is {best_fitted.params["const"]:.4f} (p = {best_fitted.pvalues["const"]:.4f}), ')
    f.write(
        f'indicating a statistically significant average daily return of {best_fitted.params["const"] * 100:.4f}%.\n')
    f.write(
        f'- The AR(1) coefficient is {best_fitted.params["ar.L1"]:.4f} with p-value = {best_fitted.pvalues["ar.L1"]:.4f}. ')
    f.write('This coefficient is marginally significant at the 10% level but not at the conventional 5% level, ')
    f.write('suggesting weak negative autocorrelation in daily returns.\n\n')

    # 3.3 Residual Diagnostics
    f.write('### 3.3 Residual Diagnostics\n\n')
    f.write('**ACF of Residuals:**\n\n')
    f.write('![Residuals ACF](figures/residuals_acf.png)\n\n')
    f.write('**Residuals Density Plot:**\n\n')
    f.write('![Residuals Density](figures/residuals_density.png)\n\n')

    f.write('**Ljung-Box Test:**\n\n')
    f.write('| Lags | LB Statistic | p-value |\n')
    f.write('|------|--------------|---------|\n')
    for idx in lb_test.index:
        p_val = lb_test.loc[idx, "lb_pvalue"]
        sig = ' *' if p_val < 0.05 else ''
        f.write(f'| {idx} | {lb_test.loc[idx, "lb_stat"]:.4f} | {p_val:.4f}{sig} |\n')

    f.write('**Interpretation:**\n\n')
    f.write('- The ACF plot shows no significant autocorrelation in residuals at the first 10 lags.\n')
    f.write('- The density plot shows that residuals have heavier tails compared to a normal distribution. ')
    f.write('This means extreme values occur more often than would be expected under a normal distribution.\n')
    f.write(f'- The Ljung-Box test shows no significant autocorrelation at lags 5 and 10 (p > 0.05), ')
    f.write(f'but indicates some remaining autocorrelation at lag 15 (p = {lb_test.loc[15, "lb_pvalue"]:.4f}). ')
    f.write('This may reflect patterns in volatility rather than in the returns themselves.\n\n')

    f.write('**Overall Assessment:**\n\n')
    f.write('The ARIMA(1,0,0) model provides a reasonable description of the return dynamics. ')
    f.write(f'The AR(1) coefficient of {best_fitted.params["ar.L1"]:.4f} suggests weak negative correlation, ')
    f.write('but it is not strong enough to be statistically significant at the 5% level. ')
    f.write('The model can be used for forecasting, though we should expect limited accuracy.\n\n')

    f.write('---\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Stage 4. Forecasting Results
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Stage 4. Forecasting Results\n\n')

    # 4.1 Forecast Plot
    f.write('### 4.1 20-Step Ahead Forecast\n\n')
    f.write('![Forecast vs Actual](figures/forecast_vs_actual.png)\n\n')

    # 4.2 Forecast Table (first 10)
    f.write('**First 10 Forecasted vs Actual Returns:**\n\n')
    f.write('| Date | Forecast | Actual |\n')
    f.write('|------|----------|--------|\n')
    for i in range(min(10, len(actual_aligned))):
        f.write(
            f'| {actual_aligned.index[i].strftime("%Y-%m-%d")} | {forecast_aligned.iloc[i]:.6f} ({forecast_aligned.iloc[i] * 100:.4f}%) | {actual_aligned.iloc[i]:.6f} ({actual_aligned.iloc[i] * 100:.4f}%) |\n')

    # 4.3 Forecast Accuracy
    f.write('\n### 4.2 Forecast Accuracy\n\n')
    f.write('| Metric | ARIMA(1,0,0) | Naive (Zero) |\n')
    f.write('|--------|--------------|--------------|\n')
    f.write(
        f'| MAE (Mean Absolute Error) | {mae:.6f} ({mae * 100:.4f}%) | {naive_mae:.6f} ({naive_mae * 100:.4f}%) |\n')
    f.write(
        f'| RMSE (Root Mean Squared Error) | {rmse:.6f} ({rmse * 100:.4f}%) | {naive_rmse:.6f} ({naive_rmse * 100:.4f}%) |\n\n')

    f.write('**Interpretation:**\n\n')
    f.write(
        f'The ARIMA(1,0,0) model achieves a MAE of {mae:.6f} ({mae * 100:.4f}%) and RMSE of {rmse:.6f} ({rmse * 100:.4f}%). ')
    f.write('Compared to simply forecasting zero every day (the naive forecast), the ARIMA model shows ')

    if mae < naive_mae:
        improvement = ((naive_mae - mae) / naive_mae) * 100
        f.write(f'a slightly better MAE ({mae:.6f} vs {naive_mae:.6f}, improvement of {improvement:.1f}%) ')
    else:
        deterioration = ((mae - naive_mae) / naive_mae) * 100
        f.write(f'a slightly worse MAE ({mae:.6f} vs {naive_mae:.6f}, {deterioration:.1f}% higher) ')

    if rmse < naive_rmse:
        improvement_rmse = ((naive_rmse - rmse) / naive_rmse) * 100
        f.write(
            f'and a slightly better RMSE ({rmse:.6f} vs {naive_rmse:.6f}, improvement of {improvement_rmse:.1f}%). ')
    else:
        deterioration_rmse = ((rmse - naive_rmse) / naive_rmse) * 100
        f.write(f'and a slightly worse RMSE ({rmse:.6f} vs {naive_rmse:.6f}, {deterioration_rmse:.1f}% higher). ')

    f.write(
        'The differences are very small, meaning the model does not provide meaningful improvement over a simple guess.\n\n')
    f.write('This makes sense given the weak autocorrelation found in the returns ')
    f.write(f'(AR coefficient of {best_fitted.params["ar.L1"]:.4f}, p = {best_fitted.pvalues["ar.L1"]:.4f}). ')
    f.write('If past returns do not strongly predict future returns, then forecasts will not be accurate.\n\n')

    f.write('---\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Conclusion
    # ------------------------------------------------------------------------------------------------------------------
    f.write('## Conclusion\n\n')
    f.write('This analysis examined the price and return dynamics of a high-tech portfolio ')
    f.write(f'over the period {df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")}. ')
    f.write('The main findings are summarized below:\n\n')

    f.write('### Summary of Findings\n\n')

    f.write('**1. Price Series Analysis**\n')
    f.write(f'- The portfolio price increased from ${price.min():.2f} to ${price.max():.2f} over the sample period.\n')
    f.write(
        f'- The series is non-stationary (ADF p-value = {adf_result[1]:.4f}), consistent with random walk behavior.\n')
    f.write('- ACF decays slowly while PACF cuts off after lag 1, confirming the presence of a unit root.\n\n')

    f.write('**2. Returns Series Analysis**\n')
    f.write(f'- Daily returns have a mean of {returns.mean():.6f} ({returns.mean() * 100:.4f}%) ')
    f.write(f'and standard deviation of {returns.std():.6f} ({returns.std() * 100:.4f}%).\n')
    f.write('- The returns series is stationary (ADF p-value ≈ 0).\n')
    f.write(f'- ACF at lag 1 is {acf_returns[1]:.4f} and PACF at lag 1 is {pacf_returns[1]:.4f}, ')
    f.write('both within the 95% confidence band, indicating no significant autocorrelation.\n\n')

    f.write('**3. ARIMA Modeling**\n')
    f.write(f'- The ARIMA(1,0,0) model was selected as the best specification (AIC = {best_result["aic"]:.2f}).\n')
    f.write(f'- The AR(1) coefficient is {best_fitted.params["ar.L1"]:.4f} (p = {best_fitted.pvalues["ar.L1"]:.4f}), ')

    if best_fitted.pvalues["ar.L1"] < 0.05:
        f.write('indicating statistically significant negative autocorrelation in daily returns.\n')
    else:
        f.write('indicating weak negative autocorrelation that is not statistically significant at the 5% level.\n')

    f.write('- Residual diagnostics: ')
    if lb_test.loc[15, "lb_pvalue"] < 0.05:
        f.write(
            f'Ljung-Box test shows some remaining autocorrelation at lag 15 (p = {lb_test.loc[15, "lb_pvalue"]:.4f}), ')
        f.write('which may reflect patterns in volatility.\n')
    else:
        f.write('Ljung-Box test confirms residuals have no significant autocorrelation (all p-values > 0.05).\n')
    f.write('- The density plot shows that residuals have heavier tails than a normal distribution ')
    f.write('(kurtosis = {:.2f}).\n\n'.format(residuals.kurtosis()))

    f.write('**4. Forecasting Performance**\n')
    f.write(
        f'- The 20-step ahead forecast yields MAE = {mae:.6f} ({mae * 100:.4f}%) and RMSE = {rmse:.6f} ({rmse * 100:.4f}%).\n')
    f.write(f'- Compared to a naive zero forecast (MAE = {naive_mae:.6f}, RMSE = {naive_rmse:.6f}), ')

    if mae < naive_mae:
        f.write('the ARIMA model shows modest improvement.\n')
    else:
        f.write('the ARIMA model performs similarly, with no significant improvement.\n')
    f.write('- This limited predictability means that past returns have little ability to predict future returns.\n\n')

    f.write('### Limitations and Future Directions\n\n')
    f.write('- The ARIMA model captures only linear dependencies in the conditional mean.\n')
    f.write('- Future work could incorporate GARCH-type models to capture volatility clustering, ')
    f.write('which is evident in the returns series (kurtosis = {:.2f}).\n'.format(residuals.kurtosis()))
    f.write('- Including macroeconomic variables or market indices (S&P 500) could improve forecast accuracy.\n')
    f.write('- Higher frequency data (intraday) or longer time horizons may reveal different patterns.\n\n')

    f.write('### Final Remarks\n\n')
    f.write(
        'The weak autocorrelation found in daily returns means that past returns have little ability to predict future returns. ')
    f.write('The ARIMA(1,0,0) model captures this weak relationship but does not produce accurate forecasts. ')
    f.write(
        'This result is expected: if returns were easy to predict, markets would quickly adjust and eliminate the predictability. ')
    f.write('Future work could explore models that focus on predicting volatility rather than returns themselves.\n\n')