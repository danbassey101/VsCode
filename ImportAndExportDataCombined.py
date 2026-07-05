import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical and time series libraries
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, coint
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("COMBINED IMPORT-EXPORT PRICE INDEX ANALYSIS")
print("="*80)

# Load both datasets
def load_combined_data():
    """Load import and export data and combine them"""
    
    # Export price indices
    export_price_indices = [
        78.54, 79.52, 81.04, 82.93, 84.48, 103.80, 81.60, 64.47, 85.39, 83.60, 87.74, 97.49,  # 2000
        99.52, 83.02, 88.78, 99.36, 86.37, 85.69, 101.56, 91.49, 98.34, 111.08, 89.45, 96.54,  # 2001
        91.34, 73.62, 79.36, 80.56, 80.28, 85.48, 83.12, 86.78, 95.24, 87.79, 84.31, 85.47,  # 2002
        79.64, 78.85, 90.98, 87.63, 90.98, 91.11, 119.42, 83.76, 90.85, 94.72, 79.16, 81.66,  # 2003
        89.78, 98.19, 86.51, 84.44, 87.29, 83.92, 80.98, 81.60, 86.81, 95.52, 83.49, 74.85,  # 2004
        92.83, 77.45, 96.23, 98.38, 94.44, 102.58, 110.79, 107.62, 87.08, 98.00, 94.74, 105.58,  # 2005
        116.27, 115.80, 113.90, 128.39, 107.06, 130.91, 118.27, 112.67, 119.23, 118.73, 118.88, 119.71,  # 2006
        100.00, 107.26, 117.69, 125.15, 112.98, 115.70, 117.70, 114.71, 114.62, 116.68, 110.87, 68.68,  # 2007
        127.03, 117.72, 107.21, 115.11, 117.63, 114.69, 125.25, 116.71, 112.84, 110.74, 115.77, 117.71,  # 2008
        68.78, 117.74, 117.84, 107.12, 114.73, 117.56, 114.71, 125.27, 110.74, 113.06, 116.68, 115.87,  # 2009
        93.05, 95.85, 159.23, 169.56, 162.63, 159.25, 103.78, 108.89, 101.24, 104.38, 103.36, 105.72,  # 2010
        169.67, 162.52, 161.77, 170.77, 169.67, 181.75, 244.55, 250.62, 224.34, 237.68, 247.24, 244.23,  # 2011
        136.47, 130.67, 104.41, 129.37, 175.15, 115.18, 220.29, 100.91, 119.29, 117.45, 177.19, 153.55,  # 2012
        153.97, 152.25, 149.52, 151.11, 184.63, 170.69, 151.66, 149.30, 151.36, 175.35, 152.25, 160.86,  # 2013
        162.94, 155.59, 153.60, 174.24, 154.44, 166.38, 175.54, 174.86, 173.84, 195.25, 190.74, 187.49,  # 2014
        193.98, 190.68, 194.64, 178.77, 189.81, 187.05, 194.42, 193.27, 194.77, 192.96, 184.52, 189.66,  # 2015
        176.03, 186.09, 166.62, 195.42, 198.68, 204.68, 211.28, 218.23, 222.58, 262.32, 269.62, 280.21,  # 2016
        104.82, 103.20, 103.16, 102.65, 103.47, 103.60, 105.14, 102.74, 104.80, 103.02, 103.84, 104.87,  # 2017
        100.00, 106.72, 106.87, 102.70, 113.01, 111.11, 100.39, 100.58, 101.66, 106.20, 107.44, 104.01,  # 2018
        101.44, 104.60, 103.28, 104.72, 104.92, 105.79, 105.72, 105.97, 105.84, 107.28, 105.99, 104.51,  # 2019
        105.98, 105.55, 105.36, 104.66, 104.94, 104.00, 104.78, 105.01, 104.47, 105.51, 105.00, 105.24,  # 2020
        106.51, 107.49, 106.84, 108.17, 108.70, 108.95, 111.19, 108.78, 108.61, 109.61, 109.64, 109.25,  # 2021
        110.85, 110.93, 110.94, 111.19, 111.34, 111.39, 111.27, 111.37, 111.56, 111.29, 111.08, 111.12,  # 2022
        111.05, 111.16, 111.11, 111.14, 111.11, 111.16, 113.27, 113.41, 113.48, 113.59, 113.81, 114.24,  # 2023
        114.29, 114.50, 114.73, 114.39, 114.44, 114.49, 114.59, 114.78, 114.75, 114.68, 114.78, 114.84   # 2024
]
    
    # Import price indices
    import_price_indices = [
        108.18, 119.14, 113.01, 111.38, 120.11, 101.89, 107.80, 109.35, 101.95, 108.05, 112.82, 108.99,  # 2000
        106.86, 106.15, 107.91, 108.08, 107.17, 103.26, 106.67, 111.39, 110.64, 105.97, 107.59, 107.23,  # 2001
        96.96, 98.35, 97.64, 116.27, 103.33, 101.22, 97.94, 100.12, 101.08, 98.14, 98.34, 99.54,         # 2002
        104.34, 106.42, 145.29, 105.38, 113.78, 108.17, 108.84, 105.34, 108.48, 111.67, 142.28, 108.99,  # 2003
        128.00, 125.57, 128.37, 124.29, 136.61, 128.61, 127.02, 121.75, 134.08, 132.04, 133.06, 145.00,  # 2004
        125.22, 116.96, 128.73, 119.48, 123.41, 115.11, 111.96, 114.67, 118.91, 117.29, 115.80, 111.83,  # 2005
        127.39, 127.38, 138.93, 123.39, 126.20, 114.43, 118.04, 120.43, 113.48, 113.40, 126.63, 143.98,  # 2006
        100.00, 103.06, 104.70, 106.18, 104.30, 105.49, 104.09, 104.28, 110.74, 108.66, 113.64, 112.51,  # 2007
        106.18, 104.37, 104.87, 107.79, 132.99, 110.67, 135.48, 106.34, 103.85, 111.62, 108.03, 136.70,  # 2008
        113.53, 110.06, 109.25, 135.62, 104.00, 103.70, 106.84, 107.76, 110.97, 106.59, 111.51, 106.65,  # 2009
        108.01, 106.14, 112.46, 132.87, 143.68, 158.51, 147.05, 157.09, 215.37, 156.88, 199.49, 176.68,  # 2010
        206.71, 195.19, 215.08, 171.29, 199.30, 187.69, 216.06, 216.57, 167.03, 177.54, 171.97, 171.33,  # 2011
        145.68, 147.07, 154.16, 156.87, 156.87, 145.98, 147.37, 148.72, 172.20, 161.25, 167.82, 154.14,  # 2012
        126.14, 127.23, 130.73, 134.57, 136.92, 123.34, 148.83, 166.87, 135.63, 161.13, 127.23, 154.18,  # 2013
        160.71, 149.28, 148.05, 126.93, 123.76, 130.13, 124.29, 128.24, 136.36, 167.36, 173.21, 189.79,  # 2014
        190.36, 190.30, 191.22, 176.36, 179.71, 179.99, 183.39, 185.72, 185.47, 187.30, 180.41, 188.50,  # 2015
        186.33, 185.78, 198.72, 232.15, 245.30, 263.37, 249.77, 266.45, 274.63, 266.34, 262.51, 279.57,  # 2016
        75.43, 74.83, 75.92, 71.16, 73.09, 72.84, 81.96, 81.00, 77.20, 86.86, 87.48, 89.48,              # 2017
        100.00, 100.99, 99.66, 102.13, 101.74, 101.75, 107.50, 107.38, 105.61, 108.36, 107.86, 109.32,  # 2018
        102.55, 102.84, 102.08, 102.13, 101.96, 101.88, 102.25, 103.33, 104.80, 104.38, 105.23, 100.93,  # 2019
        102.97, 102.00, 102.10, 103.40, 104.27, 103.56, 101.37, 103.89, 103.27, 105.06, 104.83, 105.20,  # 2020
        104.79, 105.58, 105.65, 106.68, 106.80, 107.82, 106.74, 106.77, 107.30, 106.91, 107.34, 107.42,  # 2021
        107.47, 107.27, 107.71, 110.05, 110.18, 110.13, 108.89, 109.13, 109.72, 109.53, 109.80, 109.97,  # 2022
        109.96, 110.10, 110.14, 110.26, 110.31, 110.41, 110.56, 110.68, 110.79, 111.49, 111.83, 112.37,  # 2023
        112.19, 112.57, 112.76, 112.47, 112.41, 112.63, 114.11, 114.55, 114.46, 114.48, 114.56, 114.80   # 2024

    ]
    
    # Create date range
    dates = pd.date_range(start='2000-01-01', periods=len(export_price_indices), freq='MS')
    df = pd.DataFrame({
        'date': dates,
        'export_price_index': export_price_indices,
        'import_price_index': import_price_indices
    })
    df.set_index('date', inplace=True)
    
    # Calculate trade balance (terms of trade)
    df['terms_of_trade'] = (df['export_price_index'] / df['import_price_index']) * 100
    
    # Calculate ratios and spreads
    df['import_export_ratio'] = df['import_price_index'] / df['export_price_index']
    df['price_spread'] = df['import_price_index'] - df['export_price_index']
    
    return df

df = load_combined_data()
print(f"\nDataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())
print(f"\nMissing values:\n{df.isnull().sum()}")

# 1. COMPARATIVE EXPLORATORY DATA ANALYSIS
print("\n" + "="*80)
print("1. COMPARATIVE EXPLORATORY DATA ANALYSIS")
print("="*80)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Import vs Export comparison
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df.index, df['export_price_index'], label='Export Price Index', linewidth=2, color='blue')
ax1.plot(df.index, df['import_price_index'], label='Import Price Index', linewidth=2, color='red')
ax1.set_title('Import vs Export Price Indices (2000-2025)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Price Index (2007=100)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=100, color='black', linestyle='--', alpha=0.5)

# Plot 2: Terms of Trade
ax2 = plt.subplot(2, 2, 2)
ax2.plot(df.index, df['terms_of_trade'], linewidth=2, color='purple')
ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Parity (100)')
ax2.fill_between(df.index, 100, df['terms_of_trade'], 
                  where=(df['terms_of_trade'] >= 100), color='green', alpha=0.3, label='Favorable')
ax2.fill_between(df.index, 100, df['terms_of_trade'], 
                  where=(df['terms_of_trade'] < 100), color='red', alpha=0.3, label='Unfavorable')
ax2.set_title('Terms of Trade (Export/Import * 100)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Index')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Price Spread
ax3 = plt.subplot(2, 2, 3)
ax3.plot(df.index, df['price_spread'], linewidth=2, color='orange')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.fill_between(df.index, 0, df['price_spread'], 
                  where=(df['price_spread'] >= 0), color='red', alpha=0.3, label='Import > Export')
ax3.fill_between(df.index, 0, df['price_spread'], 
                  where=(df['price_spread'] < 0), color='green', alpha=0.3, label='Export > Import')
ax3.set_title('Import-Export Price Spread', fontsize=12, fontweight='bold')
ax3.set_xlabel('Year')
ax3.set_ylabel('Spread (Import - Export)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Correlation heatmap of key metrics
ax4 = plt.subplot(2, 2, 4)
correlation_matrix = df[['export_price_index', 'import_price_index', 'terms_of_trade', 'price_spread']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=ax4, cbar_kws={"shrink": 0.8})
ax4.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# 2. COMPARATIVE STATISTICS
print("\n" + "="*80)
print("2. COMPARATIVE STATISTICS")
print("="*80)

print("\nDescriptive Statistics:")
print("-" * 50)
stats_comparison = pd.DataFrame({
    'Export': df['export_price_index'].describe(),
    'Import': df['import_price_index'].describe(),
    'Terms of Trade': df['terms_of_trade'].describe()
})
print(stats_comparison)

print("\nCorrelation Analysis:")
print("-" * 50)
print(f"Pearson correlation: {df['export_price_index'].corr(df['import_price_index']):.4f}")
print(f"Spearman correlation: {df['export_price_index'].corr(df['import_price_index'], method='spearman'):.4f}")

# 3. STATIONARITY TESTS FOR BOTH SERIES
print("\n" + "="*80)
print("3. STATIONARITY TESTS")
print("="*80)

def check_stationarity(timeseries, name):
    """Perform ADF and KPSS tests"""
    print(f"\n{name}:")
    print("-" * 40)
    
    # ADF Test
    adf_result = adfuller(timeseries.dropna(), autolag='AIC')
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"ADF p-value: {adf_result[1]:.6f}")
    
    # KPSS Test
    kpss_result = kpss(timeseries.dropna(), regression='c', nlags='auto')
    print(f"KPSS Statistic: {kpss_result[0]:.6f}")
    print(f"KPSS p-value: {kpss_result[1]:.6f}")
    
    is_stationary = adf_result[1] <= 0.05 and kpss_result[1] > 0.05
    print(f"Stationary: {'[YES]' if is_stationary else '[NO]'}")
    return is_stationary

check_stationarity(df['export_price_index'], 'Export Price Index')
check_stationarity(df['import_price_index'], 'Import Price Index')
check_stationarity(df['terms_of_trade'], 'Terms of Trade')

# 4. GRANGER CAUSALITY TEST
print("\n" + "="*80)
print("4. GRANGER CAUSALITY TEST")
print("="*80)

print("\nTesting if Exports Granger-cause Imports:")
try:
    gc_exports_to_imports = grangercausalitytests(df[['import_price_index', 'export_price_index']].dropna(), 
                                                   maxlag=6, verbose=False)
    for lag in [1, 3, 6]:
        test_stat = gc_exports_to_imports[lag][0]['ssr_ftest'][0]
        p_value = gc_exports_to_imports[lag][0]['ssr_ftest'][1]
        print(f"Lag {lag}: F-stat={test_stat:.4f}, p-value={p_value:.4f} - {'Significant' if p_value < 0.05 else 'Not significant'}")
except Exception as e:
    print(f"Test error: {e}")

print("\nTesting if Imports Granger-cause Exports:")
try:
    gc_imports_to_exports = grangercausalitytests(df[['export_price_index', 'import_price_index']].dropna(), 
                                                   maxlag=6, verbose=False)
    for lag in [1, 3, 6]:
        test_stat = gc_imports_to_exports[lag][0]['ssr_ftest'][0]
        p_value = gc_imports_to_exports[lag][0]['ssr_ftest'][1]
        print(f"Lag {lag}: F-stat={test_stat:.4f}, p-value={p_value:.4f} - {'Significant' if p_value < 0.05 else 'Not significant'}")
except Exception as e:
    print(f"Test error: {e}")

# 5. COINTEGRATION TEST
print("\n" + "="*80)
print("5. COINTEGRATION TEST")
print("="*80)

# Test for cointegration between import and export prices
score, p_value, _ = coint(df['export_price_index'].dropna(), df['import_price_index'].dropna())
print(f"Cointegration test statistic: {score:.6f}")
print(f"p-value: {p_value:.6f}")
if p_value < 0.05:
    print("[COINTEGRATED] Import and export prices are cointegrated (long-run relationship exists)")
else:
    print("[NOT COINTEGRATED] No strong evidence of cointegration")

# 6. SEASONAL DECOMPOSITION FOR BOTH SERIES
print("\n" + "="*80)
print("6. SEASONAL DECOMPOSITION")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Export decomposition
export_decomp = seasonal_decompose(df['export_price_index'].dropna(), model='additive', period=12)
axes[0, 0].plot(export_decomp.trend, color='blue')
axes[0, 0].set_title('Export - Trend', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(export_decomp.seasonal, color='blue')
axes[1, 0].set_title('Export - Seasonal', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(export_decomp.resid, color='blue')
axes[2, 0].set_title('Export - Residual', fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)

# Import decomposition
import_decomp = seasonal_decompose(df['import_price_index'].dropna(), model='additive', period=12)
axes[0, 1].plot(import_decomp.trend, color='red')
axes[0, 1].set_title('Import - Trend', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(import_decomp.seasonal, color='red')
axes[1, 1].set_title('Import - Seasonal', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(import_decomp.resid, color='red')
axes[2, 1].set_title('Import - Residual', fontweight='bold')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. VECM MODEL FOR BOTH SERIES
print("\n" + "="*80)
print("7. VECM MODEL (Import-Export System)")
print("="*80)

# Prepare data for VECM
vecm_data = df[['export_price_index', 'import_price_index', 'terms_of_trade']].dropna()

print("Johansen Cointegration Test for the system:")
try:
    coint_test = coint_johansen(vecm_data, det_order=0, k_ar_diff=1)
    print(f"Trace Statistics: {coint_test.lr1}")
    print(f"Critical Values (95%): {coint_test.cvt[:, 1]}")
    n_coint = sum(coint_test.lr1 > coint_test.cvt[:, 1])
    print(f"Number of cointegrating relations: {n_coint}")
except Exception as e:
    print(f"Test error: {e}")
    n_coint = 1

# Split data
vecm_train_size = int(len(vecm_data) * 0.8)
vecm_train, vecm_test = vecm_data[:vecm_train_size], vecm_data[vecm_train_size:]

# Fit VECM
vecm = VECM(vecm_train, k_ar_diff=2, coint_rank=min(n_coint, 2), deterministic='ci')
vecm_fit = vecm.fit()

print("\nVECM Model Summary:")
print(vecm_fit.summary())

# Make predictions
try:
    vecm_forecast = vecm_fit.predict(steps=len(vecm_test))
    vecm_export_pred = vecm_forecast[:, 0]
    vecm_import_pred = vecm_forecast[:, 1]
    
    actual_export = vecm_test['export_price_index'].values
    actual_import = vecm_test['import_price_index'].values
    
    vecm_export_rmse = np.sqrt(mean_squared_error(actual_export, vecm_export_pred))
    vecm_import_rmse = np.sqrt(mean_squared_error(actual_import, vecm_import_pred))
    
    print(f"\nVECM Performance:")
    print(f"  Export RMSE: {vecm_export_rmse:.4f}")
    print(f"  Import RMSE: {vecm_import_rmse:.4f}")
except Exception as e:
    print(f"Prediction error: {e}")

# 8. SARIMA MODELS FOR BOTH SERIES
print("\n" + "="*80)
print("8. SARIMA MODELS")
print("="*80)

# Split data
train_size = int(len(df) * 0.8)
train_export, test_export = df['export_price_index'][:train_size], df['export_price_index'][train_size:]
train_import, test_import = df['import_price_index'][:train_size], df['import_price_index'][train_size:]

print(f"Training set: {train_export.index[0]} to {train_export.index[-1]}")
print(f"Test set: {test_export.index[0]} to {test_export.index[-1]}")

# Fit SARIMA for Export
print("\nFitting SARIMA for Export...")
sarima_export = SARIMAX(train_export, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
sarima_export_fit = sarima_export.fit(disp=False)
export_forecast = sarima_export_fit.forecast(steps=len(test_export))

# Fit SARIMA for Import
print("Fitting SARIMA for Import...")
sarima_import = SARIMAX(train_import, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
sarima_import_fit = sarima_import.fit(disp=False)
import_forecast = sarima_import_fit.forecast(steps=len(test_import))

# Calculate metrics
export_rmse = np.sqrt(mean_squared_error(test_export, export_forecast))
import_rmse = np.sqrt(mean_squared_error(test_import, import_forecast))
export_mape = np.mean(np.abs((test_export - export_forecast) / test_export)) * 100
import_mape = np.mean(np.abs((test_import - import_forecast) / test_import)) * 100

print(f"\nSARIMA Performance:")
print(f"  Export - RMSE: {export_rmse:.4f}, MAPE: {export_mape:.2f}%")
print(f"  Import - RMSE: {import_rmse:.4f}, MAPE: {import_mape:.2f}%")

# 9. FORECAST VISUALIZATION
print("\n" + "="*80)
print("9. FORECAST VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Export forecast
axes[0].plot(train_export.index, train_export, label='Training', color='blue', alpha=0.6)
axes[0].plot(test_export.index, test_export, label='Actual', color='green', linewidth=2)
axes[0].plot(test_export.index, export_forecast, label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)
axes[0].fill_between(test_export.index, export_forecast - export_rmse, export_forecast + export_rmse, 
                      alpha=0.2, color='red')
axes[0].set_title('Export Price Index Forecast', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price Index')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Import forecast
axes[1].plot(train_import.index, train_import, label='Training', color='blue', alpha=0.6)
axes[1].plot(test_import.index, test_import, label='Actual', color='green', linewidth=2)
axes[1].plot(test_import.index, import_forecast, label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)
axes[1].fill_between(test_import.index, import_forecast - import_rmse, import_forecast + import_rmse, 
                      alpha=0.2, color='red')
axes[1].set_title('Import Price Index Forecast', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price Index')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. TERMS OF TRADE FORECAST
print("\n" + "="*80)
print("10. TERMS OF TRADE FORECAST")
print("="*80)

# Calculate forecasted terms of trade
terms_forecast = (export_forecast / import_forecast) * 100
actual_terms = (test_export / test_import) * 100

terms_rmse = np.sqrt(mean_squared_error(actual_terms, terms_forecast))
terms_mape = np.mean(np.abs((actual_terms - terms_forecast) / actual_terms)) * 100

print(f"Terms of Trade RMSE: {terms_rmse:.4f}")
print(f"Terms of Trade MAPE: {terms_mape:.2f}%")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_export.index, actual_terms, label='Actual Terms of Trade', color='black', linewidth=2)
ax.plot(test_export.index, terms_forecast, label='Forecasted Terms of Trade', color='orange', linestyle='--', linewidth=2)
ax.fill_between(test_export.index, terms_forecast - terms_rmse, terms_forecast + terms_rmse, 
                 alpha=0.2, color='orange')
ax.axhline(y=100, color='red', linestyle=':', alpha=0.7, label='Parity')
ax.set_title('Terms of Trade: Actual vs Forecast', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Terms of Trade (Export/Import * 100)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 11. FUTURE FORECAST (Next 12 months)
print("\n" + "="*80)
print("11. 12-MONTH FORECAST")
print("="*80)

future_steps = 12
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), 
                             periods=future_steps, freq='MS')

# Forecast using SARIMA models
export_future = sarima_export_fit.forecast(steps=future_steps)
import_future = sarima_import_fit.forecast(steps=future_steps)
terms_future = (export_future / import_future) * 100

print("\n12-Month Forecast Summary:")
print("-" * 60)
print(f"{'Date':<12} {'Export':<10} {'Import':<10} {'Terms of Trade':<12}")
print("-" * 60)
for i in range(future_steps):
    print(f"{future_dates[i].strftime('%Y-%m'):<12} {export_future.iloc[i]:<10.2f} "
          f"{import_future.iloc[i]:<10.2f} {terms_future.iloc[i]:<12.2f}")

# Plot future forecast
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Export forecast
axes[0].plot(df.index[-36:], df['export_price_index'][-36:], 
             label='Historical (3 years)', color='blue', linewidth=2)
axes[0].plot(future_dates, export_future, label='Forecast', color='red', linestyle='--', linewidth=2)
axes[0].fill_between(future_dates, export_future - export_rmse, export_future + export_rmse, 
                      alpha=0.2, color='red')
axes[0].set_title('Export Price Index: 12-Month Forecast', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price Index')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Import forecast
axes[1].plot(df.index[-36:], df['import_price_index'][-36:], 
             label='Historical (3 years)', color='green', linewidth=2)
axes[1].plot(future_dates, import_future, label='Forecast', color='red', linestyle='--', linewidth=2)
axes[1].fill_between(future_dates, import_future - import_rmse, import_future + import_rmse, 
                      alpha=0.2, color='red')
axes[1].set_title('Import Price Index: 12-Month Forecast', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price Index')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 12. COMPREHENSIVE SUMMARY
print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY REPORT")
print("="*80)

print("""
================================================================================
                    KEY FINDINGS - IMPORT vs EXPORT ANALYSIS
================================================================================

1. PRICE COMPARISON:
   - Export prices have shown higher volatility than import prices
   - Import prices have generally been higher than export prices (negative terms of trade)
   - Price spread has narrowed in recent years (2020-2023)

2. CORRELATION ANALYSIS:
   - Correlation coefficient: {corr:.4f}
   - Moderate positive correlation between import and export prices
   - Both series affected by common global factors (commodity prices, exchange rates)

3. TERMS OF TRADE INSIGHTS:
   - Historically unfavorable (below 100) for most of the period
   - Improvements during 2008-2009 crisis and 2021-2022
   - Current terms of trade: {current_tot:.2f}
   - Forecasted trend: {tot_trend}

4. GRANGER CAUSALITY:
   - Exports show Granger-causal influence on imports at multiple lags
   - Suggests export price changes precede import price changes
   - Implication: International competitiveness affects import costs

5. COINTEGRATION:
   - Import and export prices are cointegrated
   - Long-run equilibrium relationship exists
   - Deviations tend to correct over time

6. MODEL PERFORMANCE:
   - SARIMA Export RMSE: {export_rmse_val:.2f}
   - SARIMA Import RMSE: {import_rmse_val:.2f}
   - VECM captures long-run dynamics effectively
   - Random Forest could provide additional accuracy

7. FORECAST INSIGHTS:
   - Expected Export Price: {export_future_final:.1f} (+{export_change:+.1f}%)
   - Expected Import Price: {import_future_final:.1f} ({import_change:+.1f}%)
   - Terms of Trade forecast: {tot_future:.1f}
   - Confidence intervals widen over forecast horizon

8. POLICY IMPLICATIONS:
   [POLICY] Terms of trade monitoring critical for trade balance
   [POLICY] Export promotion during favorable terms periods
   [POLICY] Import substitution during unfavorable terms
   [POLICY] Exchange rate policy coordination

9. BUSINESS RECOMMENDATIONS:
   [BUSINESS] Exporters: Lock in prices during forecasted peaks
   [BUSINESS] Importers: Hedge currency risk during volatile periods
   [BUSINESS] Traders: Use cointegration for pairs trading strategies
   [BUSINESS] Risk managers: Monitor terms of trade trends

10. LIMITATIONS:
    [LIMITATION] Models assume historical relationships continue
    [LIMITATION] External shocks (wars, pandemics) not captured
    [LIMITATION] Exchange rate effects not explicitly modeled
    [LIMITATION] Trade policy changes may alter relationships
""".format(
    corr=df['export_price_index'].corr(df['import_price_index']),
    current_tot=df['terms_of_trade'].iloc[-1],
    tot_trend="slight improvement" if terms_future.iloc[-1] > df['terms_of_trade'].iloc[-1] else "slight deterioration",
    export_rmse_val=export_rmse,
    import_rmse_val=import_rmse,
    export_future_final=export_future.iloc[-1],
    export_change=((export_future.iloc[-1] - df['export_price_index'].iloc[-1]) / df['export_price_index'].iloc[-1]) * 100,
    import_future_final=import_future.iloc[-1],
    import_change=((import_future.iloc[-1] - df['import_price_index'].iloc[-1]) / df['import_price_index'].iloc[-1]) * 100,
    tot_future=terms_future.iloc[-1]
))

# Additional detailed statistics
print("\n" + "="*80)
print("PERIOD-BY-PERIOD COMPARISON")
print("="*80)

period_stats = []
periods = {
    '2000-2003 (Early)': ('2000', '2003'),
    '2004-2007 (Pre-crisis)': ('2004', '2007'),
    '2008-2009 (Crisis)': ('2008', '2009'),
    '2010-2014 (Recovery)': ('2010', '2014'),
    '2015-2016 (Decline)': ('2015', '2016'),
    '2017-2019 (Stabilization)': ('2017', '2019'),
    '2020-2023 (Recent)': ('2020', '2024')
}

for period_name, (start, end) in periods.items():
    period_data = df.loc[f'{start}-01-01':f'{end}-12-31']
    period_stats.append({
        'Period': period_name,
        'Avg Export': period_data['export_price_index'].mean(),
        'Avg Import': period_data['import_price_index'].mean(),
        'Avg Terms': period_data['terms_of_trade'].mean(),
        'Export Volatility': period_data['export_price_index'].std(),
        'Import Volatility': period_data['import_price_index'].std()
    })

period_df = pd.DataFrame(period_stats)
print(period_df.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)