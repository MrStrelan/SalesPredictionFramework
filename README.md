 Project Overview
This repository contains the codebase for a bachelor thesis project developed by Daniil Strelan, a student at the IT University of Copenhagen. The work is conducted in collaboration with a real B2B company operating in the healthcare supply chain sector. Due to confidentiality, original sales data is excluded from this repository, but all preprocessing logic and pipeline components are included.

 Project Aim
The project delivers a modular, reusable Python forecasting framework designed for weekly B2B sales prediction over a 3-month horizon. It integrates internal transactional data with external market indicators (e.g. stock movements of competitors) to support hybrid time series modeling using:

SARIMA (classical statistical model for capturing seasonality and trend)

LSTM (deep learning model for learning nonlinear patterns and residual corrections)

The pipelines are implemented with clean architecture principles—enabling data scientists and analysts to adapt and expand the framework for various forecasting experiments.

 Target Audience
This repository is intended for practitioners and researchers in data science, analytics, or forecasting who are familiar with:

Time series analysis (e.g., differencing, ACF/PACF, stationarity)

Machine learning pipelines

Model tuning, evaluation metrics (RMSE, MAE, MAPE)

Python tools such as pandas, scikit-learn, keras, and statsmodels

Each pipeline step is written with clarity and modularity in mind—so users can experiment with different data splits, model parameters, or feature inputs.

 Context and Motivation
B2B companies selling through third-party retailers often face highly variable demand due to seasonality, renewal cycles, and market signals that affect purchasing behavior. Despite increased data availability, many businesses lack structured tools to reliably forecast short-term demand.

This project aims to address that gap by:

Designing a forecasting pipeline architecture that merges multiple heterogeneous sources (internal sales, external stock trends, calendar effects)

Comparing classic SARIMA models with deep learning-based LSTM architectures

Demonstrating a hybrid model approach that improves over single-model predictions

Creating a fully documented, extensible pipeline to support future development and deployment

 Core Contributions
Modular SARIMA forecasting pipeline with model comparison, scaling, and evaluation

Residual-based LSTM correction model leveraging competitor data and calendar effects

Final hybrid forecasting model that combines SARIMA and LSTM predictions

Tools for data preprocessing, feature correlation analysis, and external API integration

Project structure enabling experimentation, tuning, and repeatable forecasting workflows


Project file structure:

.
├── Actuals/                             # Raw sales data
├── Budget/                              # Raw budget forecasts
├── tuner_dir/                           # Keras tuner checkpoint dir (optional)
├── Working_df/                          # Cleaned + merged datasets used in pipelines - files in
│   │                                        here are generated while running pipeline
│   ├── Actuals_*.csv                    # Variants of enriched actuals
│   ├── Budget_all.csv                   # Clean budget data
│   ├── Competitors.csv                  # Market stock info
│   ├── residuals.csv                    # Final residuals (SARIMA - actual)
│   ├── residuals_new.csv                # (Optional backup version)
│   └── Selected_lagged_features.csv     # Lagged competitor features
│
├── 202504 - Deployement numbers.xlsx    # Deployment reference file
├── BCI.csv                              # BCI indicator (Country-level)
│
├── .gitignore
├── best_model.keras                     # Saved trained LSTM model
│
├── FeatureAnalisis.ipynb                # Notebook for EDA/correlation
├── FetchingAPI.ipynb                    # (Optional) API integration logic
├── LoadingCSV.ipynb                     # One-time data prep script
│
├── LSTM.ipynb                           # LSTM with more granular data input
├── MainModel.ipynb                      # Full end-to-end pipeline (SARIMA, LSTM with SARIMA residuals, Hybrid)
│
└── README.md                            # Project documentation (you're here)



 Project Pipeline Overview
This project builds a hybrid sales forecasting framework combining SARIMA and LSTM, integrating both internal transactional and external market data.

Below is a step-by-step map showing how all core pipelines are connected:

1. Raw Data Preparation
Source: LoadingCSV.ipynb

Combines quarterly Actuals/Budget CSVs from folders (e.g. Actuals_2024_Qtr1.2.csv)

Cleans and enriches:

Weekly time alignment
Item code/suffix extraction
Rolling & lag features,
Subscription flags
Expected renewal logic

 Output:

Working_df/Actuals_all_enriched.csv
Working_df/Budget_all.csv

2. Market Feature Extraction
Source: FetchingAPI.ipynb

Fetches weekly stock data from Yahoo Finance (e.g. ALGN, STMN.SW)

Engineers:
Rolling means/volatility

Saves to:
Working_df/Competitors.csv

Optional:
Contains placeholders for macroeconomic data (e.g. from TradingEconomics or World Bank) for future experimentation

3. Market Feature Correlation Analysis
Source: FeatureAnalisys.ipynb

Merges Competitors.csv with actual sales
Analyzes lagged correlations between stock signals and sales
Selects best-performing lagged features

Saves:
Working_df/Selected_lagged_features.csv ← used in LSTM later

4. SARIMA Forecasting Pipeline
Source: MainModel.py

Loads enriched actuals (Actuals_all_enriched.csv)

Performs SARIMA modeling with:
Model selection (via ACF/PACF)
Seasonal tuning
Evaluation (MAE/RMSE/MAPE)
Scales and backtests results
Saves SARIMA residuals:
Working_df/residuals.csv

5. LSTM Residual Prediction Pipeline
Source: MainModel.py

Loads:
Residuals from SARIMA
Budget data
Lagged stock features
Builds LSTM model to learn SARIMA residual patterns
Trains model to correct systematic SARIMA errors
Plots & evaluates predictions

6. Hybrid Forecasting Pipeline
Source: MainModel.py

Reconstructs SARIMA baseline forecast
Adds LSTM residual correction → final hybrid prediction
Evaluates against SARIMA-only baseline
Shows improvement in RMSE, MAE

7. (Optional) Experimental Granular LSTM
Source: LSTM.ipynb

Attempts to forecast raw sales using fully encoded tabular data

Includes:
Custom LSTMDataGenerator
Log-transformation
Multi-layer LSTM training
! Not used in final pipeline due to low accuracy

Useful for future research (e.g. item- or client-level models)

 Recommended Execution Order

LoadingCSV.ipynb
FetchingAPI.ipynb
FeatureAnalisys.ipynb (saves Selected_lagged_features.csv)
MainModel.py → run SARIMA Pipeline
MainModel.py → run LSTM Residual Pipeline
MainModel.py → run Hybrid Forecast Pipeline
(Optional) LSTM.ipynb → for advanced experimentation


----------------Detailed sections overview---------------------

 How to Run: Initial Data Loading & Enrichment Pipeline
This pipeline prepares your raw quarterly actuals and budget CSVs into a cleaned, feature-enriched weekly format. This step is essential before running the SARIMA, LSTM, or hybrid pipelines — it standardizes the data format, computes lag features, encodes flags, and detects subscription renewal cycles.

All steps are executed in LoadingCSV.ipynb, and the output is saved into the Working_df/ directory.

 Input File Format & Naming Convention
Files must be placed in either the Actuals/ or Budget/ folders, and named using the following pattern:
Actuals_YYYY_QtrX.csv and Budget_YYYY_QtrX.csv

You can split quarters into parts by adding a suffix:
Budget_2024_Qtr2.1.csv and Actuals_2023_Qtr4.3.csv

 What the Pipeline Does
1. Load All Files by Type and Year
The function load_data_by_type_and_year(data_type, year, loader_fn) finds and combines all CSVs in a given folder (e.g. Budget) that match a specific year. It supports fractional quarter files like Qtr2.1.

The core loader load_data() handles:

Parsing date columns from Year, Month, Day

Converting month names to numeric format

Creating a standard Date column

You can load files using:

run_all_steps("Budget_all_enriched", "Budget", years, "Working_df/Budget_all.csv")

2. Initial Weekly Aggregation
With preprocess_weekly_sales(), the dataset is resampled to weekly granularity using week_start. It aggregates invoice values per (customer, item, week).

3. Feature Engineering
The pipeline computes several useful features:

Lag Features: 1-week and 4-week lag of invoiced sales

Rolling Averages: 4-week and 8-week rolling mean of sales

Categorical Encoding: assigns numeric codes to countries, item suffixes, and sales models

Item Suffix Parsing: extracts item_code and item_suffix from SKUs like 80240001_Renewal

4. Binary Classification Flags
Using add_sales_binary_flags(), we mark:

is_renewal: if the sale is labeled as a renewal

is_subscription: if it's part of a subscription model

is_new_sale: first-time purchases

is_sub_renewal: subscription + renewal combination

These are later used in the SARIMA and LSTM pipelines for feature modeling.

5. Subscription Renewal Cycle Detection
Using historical data, the pipeline:

Identifies repeated renewal events by customer/item

Computes renewal intervals (mean, median, mode)

Merges these intervals into the weekly dataset

Then it flags expected renewal weeks using add_expected_renewal_flag_v2(), which checks if the current week matches a past subscription pattern.

This enables the model to anticipate expected revenue "spikes" from renewals.

 Diagnostics (Optional)
For exploratory analysis, the notebook includes:

analyze_sales_model_vs_suffix() for cross-tabulating item types

plot_weekly_sales_trend() to visualize trends across years

These are useful for EDA or sanity checks, but not required for running the forecasting models.

 Output
Once complete, the pipeline:

Saves the cleaned and enriched DataFrame to:
Working_df/Budget_all.csv or Working_df/Actuals_all_enriched.csv
This can be changed when calling run_all_steps function.

These files become the core inputs to the SARIMA and LSTM pipelines later

Format is weekly with columns like:

week_start, Sum of Invoiced Amount EUR, lag_1w, roll_4w, is_subscription, weeks_since_last_renewal, etc.

 Example:
data_years = [2017, 2018, ..., 2024]
run_all_steps("Budget_all_enriched", "Budget", data_years, "Working_df/Budget_all.csv")

This will:

Load all matching Budget_YYYY_QtrX.csv files

Process them into weekly format

Save the enriched file to the Working_df folder

You can repeat the same for "Actuals" to build your sales dataset.

This loading pipeline is the first required step before running any forecasting model.
It ensures the data is aligned, cleaned, feature-rich, and ready for time series modeling.



 Pipeline: Lagged Feature Correlation Analysis
This pipeline analyzes how external competitor stock features (from Yahoo Finance) correlate with future sales behavior. It helps uncover lagged relationships between market signals and internal business metrics, guiding the selection of meaningful predictors for forecasting models. Located in file: FeatureAnalysis.ipynb

 Purpose
To understand which competitor market signals might predict sales activity with a lead time of 1–12 weeks. This helps inform the creation of lagged features used in the LSTM and hybrid forecasting pipelines.

 Inputs Required
Working_df/residuals.csv: Weekly SARIMA residuals saved from your SARIMA pipeline.

Working_df/Competitors.csv: Output from the FetchingAPI pipeline with engineered stock features.

 What the Pipeline Does
1. Load and Align
Loads residual series and weekly competitor features.

Merges them on the Date column (aligned to Mondays).

2. Generate Lagged Sales Targets
Creates 12 new columns for sales 1 to 12 weeks after each market observation.

These become the basis for computing lead-lag correlations.

3. Correlation Analysis
Computes Pearson correlation between each market feature and each future sales lag.

Outputs a full correlation matrix (correlation_df) with shape (features × lags).

4. Heatmap Visualization
Shows how predictive each feature is across different lags via seaborn heatmap.

5. Best-Lag Feature Selector
For each feature, finds the lag with the strongest absolute correlation (≥ 0.3 by default).

Outputs top_lagged_features, which are the strongest candidates for feature selection.

6. Export Selected Features
A small lag configuration is applied to generate a final feature matrix (e.g., STMN_rolling_mean_lag1).

The result is saved as:

Working_df/Selected_lagged_features.csv
This file is used directly in the LSTM residual prediction pipeline.



 Pipeline: Customer Stock Simulation from Deployment Data
This pipeline calculates a running inventory level of stock for each (Customer, Item) pair over time, based on known deployments and observed sales. It’s ideal for modeling replenishment cycles, product adoption, or stockout risk. Located in file: FeatureAnalysis_Deployment.ipynb

 Purpose
To simulate real-world product deployment and consumption behavior by estimating stock-on-hand using deployment logs and historical weekly sales.

 Inputs Required
202504 - Deployement numbers.xlsx: A deployment log from CRM, formatted with:

MonthDay_WithYear, Customer.Customer Code, item_code, SumYTD
(Skip first 6 rows; multiple quarters/regions allowed) it skip for our specific use case and formating of xlsx file - this might get skipped depending on your data input

Working_df/Actuals_all_enriched.csv: Your cleaned and enriched actuals dataset (from the loading pipeline).

 What the Pipeline Does
1. Load Deployment Excel
Cleans and converts the deployment sheet into a usable format.

Drops unused region info and converts the MonthDay_WithYear column to deployment_date.

2. Simulate Customer Stock
Iterates through each (Customer, Item) pair chronologically.

Tracks product deployments and offsets them with units sold weekly.

Computes a new customer_stock column representing estimated inventory over time.

3. Save Result
The enriched dataset is saved as:

Working_df/Actuals_with_stock.csv
This version of the sales data includes a customer_stock column for each row and can be used as a feature in any model that predicts demand, reorder timing, or product lifecycle behavior.

4. Diagnostics
Prints the total number of unique customer-item combinations to verify tracking coverage:

Unique (customer, item) combinations: ####

 Use Cases
Understand which clinics are running low on stock and likely to reorder.

Model reorder probability as a function of customer_stock.

Add inventory-informed features into hybrid or classification-based models.



 How to Run: Competitor Stock Data Fetching & Feature Engineering Pipeline
This pipeline fetches weekly stock price data from Yahoo Finance using the yfinance API, transforms it into a clean and aligned feature matrix, and saves it for use in your time series forecasting models. It’s designed to help capture external market signals and competitor trends that may influence sales performance. The file should be ran after "Yahoo tickers" markdown

All steps are executed inside FetchingAPI.ipynb.

 Purpose
Stock movements of companies in the same market (e.g., medical device competitors) may signal macro trends that affect business-to-business ordering behavior. This pipeline:

Pulls 10 years of weekly close price and volume data,

Engineers percent changes and rolling averages,

Aligns the output by week start (Monday) so it matches your sales dataset,

Outputs a fully formatted CSV that can be joined into your main feature matrix.

 Inputs Required
This pipeline requires an internet connection and the yfinance package to fetch data directly from Yahoo Finance.
You do not need to upload any files manually — it downloads data on demand.

The only manual input is the list of stock tickers:
tickers = ['STMN.SW', 'ALGN', 'XRAY', 'HSIC', '^VIX']

You can change these tickers based on your industry competitors or leading indices. Yahoo tickers can be searched at finance.yahoo.com.

 What This Pipeline Does
1. Download Weekly Stock Data
Using yfinance.download(), the script fetches:

Adjusted close prices

Volumes
for each ticker, using a 1wk interval and a 10y history window (you can modify this as needed).

2. Process and Engineer Features
The function process_stock_data():

Cleans the raw ticker data

Calculates weekly percentage change

Adds 3-week rolling averages and rolling volatility

Drops raw close/volume columns if not needed

Aligns all rows to Mondays for merging with sales

Only engineered features are retained by default.

3. Round and Format
The helper round_features() cleans up the numeric precision:

Percentage change: 4 decimals

Rolling mean: 3 decimals

Rolling std: 6 decimals

This ensures clean file output and prevents unnecessary precision artifacts.

4. Save to CSV
The final matrix is saved as:

Working_df/Competitors.csv

Each row corresponds to the Monday of a week, and can be joined into your main sales or residual dataset via a left join on Date.

 Example Output Columns:
 Date, ALGN_pct_change, ALGN_rolling_mean, ALGN_rolling_vol, XRAY_pct_change, ...

You will get one set of engineered features per ticker. These become part of the external features used in the LSTM or hybrid forecasting pipelines.

 To Merge With Sales Data
Once saved, you can use this in your main feature engineering step:

combined = combined.join(stock_df, how='left')

Make sure both datasets are aligned on a Date or week_start index set to Mondays (asfreq('W-MON')).

This pipeline runs independently and should be re-run periodically if you want to refresh market data or test new tickers.
It's particularly useful when evaluating the influence of market sentiment or peer performance on your internal sales trends.



 Future Data Sources for Macro-Level Forecasting (Experimental Blocks)
In early exploratory stages, additional cells were tested to pull macro-economic indicators and external datasets that may influence B2B sales trends. These were not used in the current model pipelines due to lack of weekly granularity or relevance in early experiments, but they remain valuable for future expansion.

Below is a summary of those external data fetch attempts and their possible future roles:

 tradingeconomics API – Real-time Global Macro Data
The tradingeconomics library was used to fetch historical and live updates of macro indicators such as GDP, interest rates, and production indices:

te.getHistoricalData() was used to query GDP for specific countries like Mexico.

te.getHistoricalUpdates() returns the latest revisions and real-time updates.

te.getIndicatorData() and te.getAllCountries() list available indicators and regions.

Potential future use:

Inject real-time macro indicators into forecasting models

Experiment with country-specific leading indicators (GDP, PMI, inflation)

Add features for FX-sensitive markets or export regions

 These indicators are often monthly or quarterly, so they may require forward-filling or rolling averaging to align with weekly models.

 wbdata (World Bank API) – Economic Trends at Scale
The wbdata library was explored to retrieve historical data from the World Bank across regions:

Tested interest rates (FR.INR.RINR) for USA, France, Germany, and Japan.

Used wbdata.get_sources() and get_indicators() to inspect the data catalog.

Potential future use:

Global leading indicators for strategic markets

Cross-validation of economic trends between sales territories

Use in causal modeling (e.g., Granger causality testing)

 BCI (Business Confidence Index) CSV Loader
The notebook also included early parsing of a BCI file (BCI.csv), a static dataset containing monthly confidence index values by country.

Columns were renamed, and dates were parsed to datetime format.

Filtering and preview were done for entries like 'Denmark'.

Potential future use:

Use BCI as a signal for purchasing behavior expectations

Feature engineering from BCI slope, month-over-month delta, or historical deviation

These blocks are not currently integrated in any forecasting pipeline.
They remain useful for future extensions involving macro-driven forecasting,
policy-sensitive markets, or cross-border business cycle analysis.




How to Run: SARIMA Forecasting Pipeline
The SARIMA pipeline is the first stage of this forecasting framework. It models the seasonal and linear components of historical weekly sales data and produces a baseline forecast. It also calculates the residuals (the part SARIMA couldn’t explain), which are later passed into the LSTM model for hybrid modeling.

All steps below are executed in MainModel.ipynb.

Input Format
We use the file Working_df/Actuals_all_enriched.csv as input. This CSV must include at minimum:

week_start: weekly timestamp (must be aligned to Mondays),

Sum of Invoiced Amount EUR: the sales amount to forecast.

Other columns like is_renewal, weeks_since_last_renewal, is_subscription etc. are used later for LSTM feature engineering, so they can remain in the file even if unused at this stage.

You can use a different file as long as it includes a weekly date column (week_start) and a numeric target column (like sales revenue). Adjust the path and column name in the pipeline if needed.

What This Pipeline Does
The pipeline begins by loading and aggregating weekly sales, then splits the dataset into training and test sets. It plots the series, performs a stationarity test (ADF), and visualizes autocorrelations (ACF and PACF) to guide model selection.

We first fit a simple SARIMA model using one configuration: order=(1,1,1) and seasonal_order=(1,1,0,4), and generate a 12-week forecast to check performance visually.

To validate model performance, we retrain SARIMA using only the train_series (data before 2024-09-01), then forecast the same number of steps as the test set (Sep–Dec 2024). We compare forecast vs. actuals and evaluate accuracy using RMSE and MAE.

Seasonality Tuning
The pipeline then compares two seasonal cycle options (5-week and 13-week) to explore whether monthly or quarterly patterns fit better. After that, it evaluates several advanced parameter combinations (ARIMA and seasonal orders) using the compare_advanced_sarima_models function.

! Manual step required: After comparing outputs and evaluation metrics, you must manually select the best model configuration. In this pipeline, we use the “Seasonal MA+” variant:
order=(1,1,1) and seasonal_order=(1,1,2,13).

Forecast Adjustment (Scaling) (Optional)
SARIMA forecasts often under- or over-estimate total sales volume. To correct this, the pipeline includes both direct scaling and progressive scaling methods. The selected SARIMA forecast is scaled to match the mean of actuals. This step was applied after tandancy of underscaling was noticed in the SARIMA oredictions - it was checked and worked for all data periods as improving predictions scaler, but if in your data SARIMA doesn't underscale the values it might be not needed. Also this step might considered overfiting specific dataset - use wisley with your responsibility.

After scaling, the pipeline re-evaluates performance using MAE, RMSE, and MAPE.

Backtesting on Earlier Time Period
The pipeline includes a second backtesting window from March to June 2024, to verify the model's robustness across different time frames. This uses the same SARIMA configuration and progressive scaling logic. Change input date to check possible overfitting by different periods

 Final Residual Output
The most important output from this pipeline is the residuals — the unexplained component of the forecast. These are calculated as:

residuals = weekly_sales - progressively_scaled_forecast
The residuals are saved to Working_df/residuals_new.csv and will be used as the target for LSTM training in the next section of the pipeline.

At this point, you have a baseline forecast from SARIMA and cleaned residuals ready for the LSTM model.


 How to Run: LSTM Residual Forecasting Pipeline
This section of the pipeline builds an LSTM model that learns to predict the SARIMA residuals, using a wide range of enriched features. The goal is to model complex nonlinear relationships and short-term fluctuations that SARIMA cannot capture.

All steps below are executed inside MainModel.ipynb.

 Inputs Required
Working_df/residuals.csv: residuals produced and saved by the SARIMA pipeline.

Working_df/Actuals_all_enriched.csv: the enriched actuals dataset containing all time-based and categorical inputs.

Working_df/Budget_all.csv or Working_df/Actuals_all_enriched.csv: aggregated weekly budgeted sales/quantities.

Working_df/Selected_lagged_features.csv: competitor features (lagged indicators, stock info, etc.).

These should be pre-generated from the previous pipeline stage or data prep notebooks.

 What This Pipeline Does
1. Load Residuals and Enriched Sales Data
We start by loading SARIMA residuals and the enriched actuals file. The residual_series will become the target variable for this LSTM.

2. Load Budget Data
Loads weekly budget information and prepares it using load_weekly_budget. This data is merged into the LSTM feature set later.

3. Generate Weekly Features
Using create_weekly_features, we aggregate enriched features to the week level and attach SARIMA residuals as a column.
We also create lagged residuals using add_residual_lags, which helps the LSTM model learn temporal patterns.

4. Plot Residual Series
Diagnostic plots show the structure, distribution, and autocorrelation of the SARIMA residuals to ensure they are suitable for modeling.

5. Load External Features
Competitor stock or market indicators are loaded using load_weekly_competitor_features. These are assumed to already be lagged and aligned to weekly data (starting Mondays).

6. Create Calendar and Sales Aggregates
Using create_weekly_calendar_features, we extract features like week, month, quarter, is_year_end, etc. which help capture temporal effects in sales.

7. Combine All Features
With combine_features_with_residuals, we merge:

weekly aggregates,

SARIMA residuals,

competitor signals,
and then apply add_budget_features to inject budget-related inputs.

This forms the complete dataset used to train the LSTM.

8. Explore Correlations
Optional step: plot_feature_correlation() and inspect_features() help evaluate feature relevance to residuals. This is mostly diagnostic.

 LSTM Modeling
9. Feature and Target Scaling
We scale the full feature matrix (X) and residual target (y) to [0, 1] using MinMaxScaler. This is crucial for stable LSTM training.

10. Sequence Formatting
We use create_lstm_sequences to prepare data in the shape required by LSTM:
→ (samples, timesteps, features)
Here we use a window of 26 weeks (half-year) to predict the residual one week ahead.

11. Train/Test Split
The sequences are split into training and test sets using split_lstm_data. The target (y_train, y_test) is also scaled again with scale_target_minmax.

You can change the train_ratio or timesteps depending on your forecasting needs.

 (Optional) Grid Search for Hyperparameters
You can activate hyperparameter tuning by uncommenting the param_grid block and running the grid_search_lstm() function.

This performs an exhaustive search over:

LSTM unit sizes

dropout rate

dense layer size

learning rate

The best configuration (lowest validation MAE) is extracted and passed to build_and_train_stacked_lstm.

! This can take time (2-10 hours depending on the grid). Only needed once unless retraining.

 Train the LSTM
We build a stacked LSTM with:

2 LSTM layers

Dropout

Dense + ReLU output

Final dense layer (1 neuron for regression)

If param_grid is set, the model will use the best found hyperparameters. If not, it uses default values.

Training includes early stopping and validation split to avoid overfitting. Progress is tracked in history.

 Evaluate the Model
After training:

Predictions are made using model.predict()

These are inverse-transformed using inverse_scale_predictions() to get values in euros

Final plots include:

plot_lstm_predictions: shows actual vs predicted residuals

plot_training_history: shows training vs validation loss

plot_training_mae: shows MAE trend over epochs

 What the Model Learns
The LSTM is trained not to predict full sales, but the residual error from SARIMA. This makes it ideal for:

capturing nonlinear effects,

accounting for promotions, seasonality transitions,

and using external features (e.g., budget, stock market data).

In the next pipeline, we will add this predicted residual back to the SARIMA forecast to create a full hybrid sales prediction.



 How to Run: Hybrid SARIMA + LSTM Forecast Pipeline
This is the final stage of the pipeline where we combine the SARIMA base forecast with the LSTM-predicted residuals to generate the most accurate weekly sales forecast. This hybrid model takes advantage of both time series structure (via SARIMA) and nonlinear relationships (via LSTM).

All steps are executed inside MainModel.ipynb.

 What This Pipeline Does
We begin by fitting SARIMA on the entire sales history (weekly_agg['Sum of Invoiced Amount EUR']) and generating a forecast from the same point that LSTM sequences begin (defined by timesteps, usually 26 weeks). This is done using the generate_sarima_forecast_series() function, which also merges the forecast into the combined dataframe used earlier in the LSTM pipeline.

Then, we extract the final portion of the SARIMA forecast using extract_sarima_segment() — this ensures that the SARIMA output and LSTM predictions align exactly in time.

Next, the hybrid forecast is created using combine_sarima_lstm_forecast(). This function simply adds the SARIMA forecast and the predicted residuals from the LSTM, which gives us a corrected, nonlinear-enhanced sales forecast.

To evaluate the hybrid forecast, we extract the actual sales values and their corresponding dates from the combined dataframe. These are used for plotting and evaluation.

The hybrid forecast is visualized in two ways:

A direct comparison of actual sales vs. hybrid forecast

A 3-line comparison of actual sales, SARIMA-only forecast, and the hybrid forecast — to clearly show where the hybrid adds value

Finally, we evaluate both the SARIMA forecast and the hybrid forecast using evaluate_and_compare_forecasts(). This function calculates standard metrics (MAE, RMSE, MAPE) and prints a comparison of both models. It also prints the percentage improvement of RMSE for the hybrid model over SARIMA.

 What Can Be Changed
If you're adapting this pipeline for your own data:

Change the order and seasonal_order passed to generate_sarima_forecast_series() to match your chosen SARIMA configuration.

If you're using different features or sequence lengths, ensure timesteps is correctly aligned with the start of your LSTM predictions.

If your LSTM model is trained on a different target (e.g., normalized residuals), make sure to inverse transform predictions before combining with SARIMA.

 Final Output
At the end of this pipeline, you'll have:

A complete, week-by-week sales forecast using the hybrid model

A visual comparison of actual vs. forecast (SARIMA and Hybrid)

Evaluation metrics for both models

Confirmation that the hybrid forecast improves performance over SARIMA alone

This is the production-ready model output, and can be extended to future dates or adapted to rolling forecasts.



 Future Pipeline Optimization Utilities
In addition to the functions used directly in the current SARIMA, LSTM, and Hybrid pipelines, several helper functions were created to support future modularity, reusability, and experimentation. These are not currently invoked in MainModel.ipynb, but are already implemented and ready for future use cases such as refactoring, automation, or scheduled batch forecasting.

Here is a summary of those functions and their intended purposes:

 compute_sarima_residuals(series, model)
Computes the in-sample forecast from a fitted SARIMA model and returns the residuals (actual - predicted).

Intended to simplify and isolate the residual generation process for cleaner pipeline control.

Useful when exporting multiple versions of residuals for comparison or running batch residual generation across product segments.

 save_residuals(residual_series, output_path)
Saves a residual time series to disk as a .csv file.

Encapsulates a standard saving step with proper formatting (index=True, header=True).

Ideal for situations where residual saving is modularized (e.g. different residual variants, intermediate checkpoints).

 plot_lstm_residual_comparison(y_true, y_pred, title)
Plots actual vs. predicted residuals using LSTM output, for direct visual error comparison.

Mirrors plot_lstm_predictions but is targeted specifically at residual-based modeling instead of direct sales prediction.

Useful when testing multiple residual modeling strategies (e.g., LSTM, XGBoost, GRU) and needing consistent error visualization.

 prepare_hybrid_model_data(combined, target_col, sarima_forecast_series)
Builds a clean set of inputs and target (y) for hybrid model training.

Removes columns like Sum of Invoiced Amount EUR, inserts SARIMA forecast, and ensures proper alignment.

Intended for future use when retraining hybrid models on direct sales forecasts, not just residuals.

 Example: You could replace LSTM residual modeling with direct sales modeling using this function to
 generate input matrices aligned with SARIMA baseline predictions.



 Experimental Pipeline: Granular LSTM Forecasting on Raw Sales
! Note: This pipeline is experimental and was not used in the final hybrid model due to inconsistent performance. However, it is included in the project as a valuable foundation for future research, particularly for item-level, customer-level, or highly granular LSTM modeling. Location in file: LSTM.ipynb

 Purpose
This notebook explores using LSTM neural networks to forecast raw sales (Sum of Invoiced Amount EUR) with fine temporal resolution. It incorporates rich engineered features including:

Internal sales data (Actuals_all_enriched.csv)

Competitor market signals (Competitors.csv)

SARIMA residuals (residuals.csv)

Although promising in structure, the trained LSTM failed to generalize well across validation data, suggesting a need for further feature filtering, dimensionality reduction, or grouping strategies (e.g., by customer cluster, region, or product type).

 Input Files (Expected in Working_df/)
Actuals_all_enriched.csv – main weekly sales dataset (generated via LoadingCSV.ipynb)

Competitors.csv – processed Yahoo Finance stock features (FetchingAPI.ipynb)

residuals.csv – SARIMA residuals aligned to weekly dates

 What the Pipeline Does
1. Load and Merge Datasets
Uses helper functions to load and align actuals, competitor signals, and residuals on date.

2. Preprocessing
Drops unnecessary columns (e.g. country names)

Applies label encoding for categorical features

Applies z-score scaling (via StandardScaler) to numerical columns

Drops rows with missing or infinite values post-scaling

3. Signed Log Transformation
Transforms the target (Sum of Invoiced Amount EUR) to stabilize variance and reduce outliers

Ensures that predictions can be inversely transformed safely

4. Data Generator Setup
Custom LSTMDataGenerator builds 3D tensors with configurable:

Input window (e.g., 8 weeks of past data)

Forecast horizon (e.g., 1-week ahead)

Batch size

5. Train/Test Split
Chronologically splits data into pre-2023 for training and 2023+ for validation

6. Model Building
build_lstm_model() creates a stacked LSTM → LSTM → Dense architecture with dropout and Adam optimizer

Supports manual or hyperparameter-tuned configuration

7. Training
Uses EarlyStopping and ModelCheckpoint

Trains on training generator, evaluates on validation generator

8. Evaluation
evaluate_model() compares true vs predicted log-sales (converted back to EUR via inverse transform)

Outputs:

RMSE, MAE, MAPE

Line plot of actual vs predicted values

9. Hyperparameter Tuning (Optional)
Uses Keras Tuner (RandomSearch) for:

LSTM unit counts

Dropout rates

Dense layer size

Learning rate

Top models and hyperparameters can be retrieved for re-training

10. Visualizations
Validation error vs. LSTM units

Full training history (loss, val_loss)

Original sales trend over time

 Important Observations
The model architecture and data pipeline are solid, but:

The full feature set is very high-dimensional.

Some categorical groupings may be too sparse for neural models.

The model likely suffers from underfitting or overfitting depending on configuration.

 Future Use Recommendations
If revisiting this approach, consider:

Training separate models for subsets (e.g., per sales model, region, top clients)

Dimensionality reduction (e.g., PCA, feature selection based on correlation)

Time-aware embeddings for Customer.Customer Code or item_code

Using attention or transformer layers instead of stacked LSTM

 Key Functions for Reuse
run_data_loading_pipeline() – merges actuals + competitors + residuals

run_preprocessing_pipeline() – handles full categorical + numerical preprocessing

run_lstm_training_pipeline() – executes model training with selected architecture

evaluate_model() – generates forecast vs actual plots and error metrics

best_model_hyperparameters() – (optional) Keras Tuner for automatic tuning

 Output
There are no saved predictions in this experimental pipeline. 
Instead, it visualizes output and prints error metrics during evaluation. You can export final results manually if desired.

