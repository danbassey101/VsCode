# Nigeria Import & Export Price Index Analysis

## 📊 Project Overview

This project analyzes the relationship between **Nigeria's import and export price indices** from **2000 to 2024** using statistical analysis, time-series techniques, econometric testing, and forecasting.

The analysis investigates how import and export prices have evolved over time, whether a long-term relationship exists between them, how changes in one series influence the other, and what future trends may look like.

A key focus of the project is the **Terms of Trade (ToT)**, calculated as:

> **Terms of Trade = (Export Price Index / Import Price Index) × 100**

The project combines exploratory data analysis with advanced time-series and econometric methods including **ADF, KPSS, Granger causality, cointegration, VECM, seasonal decomposition, and SARIMA forecasting**.

---

## 🎯 Objectives

The major objectives of this project are to:

* Analyze historical movements in Nigeria's import and export price indices.
* Compare import and export price trends over time.
* Calculate and evaluate Nigeria's Terms of Trade.
* Examine the correlation between import and export prices.
* Test whether the time series are stationary.
* Determine whether import and export prices have a long-run relationship.
* Investigate directional relationships using Granger causality.
* Identify trend and seasonal patterns.
* Model the relationship between the variables using VECM.
* Forecast future import and export price indices using SARIMA.
* Forecast future Terms of Trade.
* Generate insights that can support trade policy and business decision-making.

---

## 📁 Dataset

The dataset contains monthly observations of:

| Variable              | Description                                        |
| --------------------- | -------------------------------------------------- |
| `date`                | Monthly observation date                           |
| `export_price_index`  | Export price index                                 |
| `import_price_index`  | Import price index                                 |
| `terms_of_trade`      | Export price index relative to import price index  |
| `import_export_ratio` | Ratio of import prices to export prices            |
| `price_spread`        | Difference between import and export price indices |

### Data Coverage

* **Frequency:** Monthly
* **Period:** 2000–2024
* **Observations:** 300 monthly observations
* **Primary indicators:** Import Price Index, Export Price Index, Terms of Trade

---

## 🛠️ Technologies & Libraries

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Statistical & Econometric Analysis

* Statsmodels
* SciPy

### Machine Learning

* Scikit-learn

### Time-Series Models

* SARIMA
* VECM

---

## 🔬 Methodology

The project follows a structured analytical workflow.

### 1. Data Preparation

The import and export price-index datasets are combined into a single time-series DataFrame.

Additional analytical variables are created:

```python
df['terms_of_trade'] = (
    df['export_price_index'] / df['import_price_index']
) * 100

df['import_export_ratio'] = (
    df['import_price_index'] / df['export_price_index']
)

df['price_spread'] = (
    df['import_price_index'] - df['export_price_index']
)
```

---

### 2. Exploratory Data Analysis

The project examines:

* Import vs. export price movements
* Terms of Trade
* Import-export price spread
* Correlation between key indicators
* Descriptive statistics
* Period-by-period changes
* Historical volatility

Visualizations include time-series plots, correlation heatmaps, and comparative charts.

---

### 3. Stationarity Testing

Two statistical tests are applied:

* **Augmented Dickey-Fuller (ADF) Test**
* **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**

These tests determine whether the time series are stationary and help guide the appropriate modeling approach.

---

### 4. Granger Causality

Granger causality tests are performed with lags up to six months to investigate whether historical movements in one price index provide useful information for predicting the other.

The analysis tests both directions:

* Export prices → Import prices
* Import prices → Export prices

---

### 5. Cointegration Analysis

The **Engle-Granger cointegration test** is used to investigate whether import and export prices share a long-run equilibrium relationship.

This is important because two non-stationary economic variables may still move together over the long term.

---

### 6. Seasonal Decomposition

Both import and export price indices are decomposed into:

* Trend
* Seasonal component
* Residual component

A 12-month seasonal period is used because the dataset has monthly frequency.

---

### 7. Vector Error Correction Model (VECM)

A VECM is implemented to analyze both short-run dynamics and long-run equilibrium relationships within the import-export system.

The model uses Johansen's cointegration methodology to identify potential cointegrating relationships.

---

### 8. SARIMA Forecasting

Separate SARIMA models are fitted to:

* Export Price Index
* Import Price Index

The model specification used is:

```text
SARIMA(1,1,1)(1,1,1,12)
```

The models are evaluated using:

* RMSE
* MAPE

The resulting forecasts are then used to calculate future Terms of Trade.

---

## 📈 Key Analysis Areas

### Import vs Export Prices

The project compares the historical behavior of import and export prices to identify periods of:

* High volatility
* Price convergence
* Price divergence
* Structural changes

### Terms of Trade

A Terms of Trade value:

* **Above 100** → export prices are relatively higher than import prices.
* **Below 100** → import prices are relatively higher than export prices.
* **Equal to 100** → export and import prices are at parity.

This provides an important indicator of Nigeria's relative trading position.

---

## 🔮 Forecasting

The project generates:

### Historical/Test Forecasts

SARIMA forecasts are compared against actual observations using:

* RMSE
* MAPE

### 12-Month Forecast

The final SARIMA models are used to generate a **12-month forecast** for:

* Export Price Index
* Import Price Index
* Terms of Trade

The forecasts provide an indication of potential short-term movements based on historical patterns.

---

## 📊 Model Evaluation

The models are evaluated using:

### Root Mean Squared Error (RMSE)

RMSE measures the average magnitude of prediction errors while giving greater weight to larger errors.

### Mean Absolute Percentage Error (MAPE)

MAPE measures prediction accuracy as a percentage, making it easier to interpret forecasting performance.

---

## 💡 Business & Economic Insights

The analysis can provide useful insights for several stakeholders.

### Government & Policymakers

* Monitor changes in Nigeria's Terms of Trade.
* Support export diversification strategies.
* Identify periods of unfavorable trade conditions.
* Improve trade and exchange-rate policy coordination.
* Develop appropriate import-substitution strategies.

### Exporters

* Monitor export price trends.
* Identify periods of potentially favorable international pricing.
* Improve pricing and production decisions.

### Importers

* Monitor import-price volatility.
* Anticipate potential increases in import costs.
* Incorporate forecasts into procurement planning.

### Financial & Risk Analysts

* Monitor price relationships and volatility.
* Use time-series forecasts for scenario analysis.
* Evaluate potential exposure to international price movements.

---

## ⚠️ Limitations

The analysis has several limitations:

1. Historical relationships may not continue indefinitely.
2. External shocks such as geopolitical conflicts and global economic crises may affect future prices.
3. Exchange-rate movements are not explicitly modeled.
4. Changes in Nigerian trade policies may alter historical relationships.
5. Commodity-specific differences are not separately modeled.
6. SARIMA forecasts are primarily based on historical time-series patterns.
7. Forecast uncertainty increases as the prediction horizon increases.

---

## 🚀 Future Improvements

Future versions of the project could include:

* Incorporating Nigeria's exchange-rate data.
* Adding crude oil and commodity price variables.
* Including inflation and interest-rate indicators.
* Applying VAR/VECM models with additional macroeconomic variables.
* Testing Prophet, XGBoost, Random Forest, and other forecasting approaches.
* Performing rolling-window and walk-forward validation.
* Building an interactive **Power BI or Looker Studio dashboard**.
* Creating an automated forecasting pipeline.
* Developing scenario-based trade forecasts.

---

## 📂 Suggested Repository Structure

```text
nigeria-import-export-analysis/
│
├── data/
│   └── import_export_price_indices.csv
│
├── notebooks/
│   └── import_export_analysis.ipynb
│
├── src/
│   └── analysis.py
│
├── visualizations/
│   ├── import_export_trends.png
│   ├── terms_of_trade.png
│   ├── correlation_heatmap.png
│   └── forecasts.png
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nigeria-import-export-analysis.git
cd nigeria-import-export-analysis
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy
```

### 3. Run the analysis

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then run:

```text
notebooks/import_export_analysis.ipynb
```

---

## 📌 Conclusion

This project provides a comprehensive time-series analysis of Nigeria's import and export price indices between **2000 and 2024**.

By combining **exploratory data analysis, statistical testing, econometric modeling, seasonal decomposition, cointegration analysis, VECM, and SARIMA forecasting**, the project demonstrates how historical economic data can be transformed into actionable insights.

The analysis highlights the importance of monitoring import and export price movements and the **Terms of Trade** when evaluating Nigeria's external trade position.

---

## 👨‍💻 Author

**Daniel Bassey**

Data Analyst | Cloud Data Analyst | Business Analyst

**Skills demonstrated:** Python • Pandas • NumPy • SQL • Statistical Analysis • Time-Series Analysis • Econometrics • Data Visualization • Forecasting • Power BI • Looker Studio

---

## ⭐ If you find this project useful

Feel free to **star ⭐ the repository**, explore the analysis, and connect with me to discuss data analytics, econometrics, forecasting, and business intelligence.
