"""
Insurance Claim Prediction for DSC MCF ITB Competition
Target: Predict Claim Frequency, Severity, and Total Claim for Aug-Dec 2025
Evaluation Metric: MAPE (Mean Absolute Percentage Error)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================
print("Loading data...")

# Load policy data
df_polis = pd.read_csv('Data_Polis.csv')
print(f"Policy data loaded: {df_polis.shape[0]} records")

# Load claim data
df_klaim = pd.read_csv('Data_Klaim (1).csv')
print(f"Claim data loaded: {df_klaim.shape[0]} records")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\nPreprocessing data...")

# Convert date columns in policy data
df_polis['Tanggal Lahir'] = pd.to_datetime(df_polis['Tanggal Lahir'], format='%Y%m%d')
df_polis['Tanggal Efektif Polis'] = pd.to_datetime(df_polis['Tanggal Efektif Polis'], format='%Y%m%d')

# Calculate age at policy effective date
df_polis['Umur'] = (df_polis['Tanggal Efektif Polis'] - df_polis['Tanggal Lahir']).dt.days / 365.25

# Encode categorical variables
le_gender = LabelEncoder()
df_polis['Gender_Encoded'] = le_gender.fit_transform(df_polis['Gender'])

le_plan = LabelEncoder()
df_polis['Plan_Code_Encoded'] = le_plan.fit_transform(df_polis['Plan Code'])

le_domisili = LabelEncoder()
df_polis['Domisili_Encoded'] = le_domisili.fit_transform(df_polis['Domisili'])

# Plan code mapping for region coverage
plan_region_map = {
    'M-001': 'Worldwide',
    'M-002': 'Asia',
    'M-003': 'Domestic'
}
df_polis['Region_Coverage'] = df_polis['Plan Code'].map(plan_region_map)

# Convert date columns in claim data
df_klaim['Tanggal Pasien Masuk RS'] = pd.to_datetime(df_klaim['Tanggal Pasien Masuk RS'])
df_klaim['Tanggal Pasien Keluar RS'] = pd.to_datetime(df_klaim['Tanggal Pasien Keluar RS'])
df_klaim['Tanggal Pembayaran Klaim'] = pd.to_datetime(df_klaim['Tanggal Pembayaran Klaim'], errors='coerce')

# Create claim month column (using entry date)
df_klaim['Claim_Month'] = df_klaim['Tanggal Pasien Masuk RS'].dt.to_period('M')
df_klaim['Claim_Year'] = df_klaim['Tanggal Pasien Masuk RS'].dt.year
df_klaim['Claim_Month_Num'] = df_klaim['Tanggal Pasien Masuk RS'].dt.month

# Calculate length of stay
df_klaim['Length_of_Stay'] = (df_klaim['Tanggal Pasien Keluar RS'] - df_klaim['Tanggal Pasien Masuk RS']).dt.days

# Encode claim type
df_klaim['Reimburse_Cashless_Encoded'] = (df_klaim['Reimburse/Cashless'] == 'C').astype(int)
df_klaim['Inpatient_Encoded'] = (df_klaim['Inpatient/Outpatient'] == 'IP').astype(int)

# Extract ICD category (first character)
df_klaim['ICD_Category'] = df_klaim['ICD Diagnosis'].astype(str).str[0]

# ============================================================
# 3. FEATURE ENGINEERING - POLICY LEVEL
# ============================================================
print("\nEngineering policy features...")

# Aggregate claims per policy
policy_claims = df_klaim.groupby('Nomor Polis').agg({
    'Claim ID': 'count',
    'Nominal Klaim Yang Disetujui': ['sum', 'mean', 'std', 'min', 'max'],
    'Nominal Biaya RS Yang Terjadi': ['sum', 'mean'],
    'Length_of_Stay': ['sum', 'mean'],
    'Inpatient_Encoded': 'mean',
    'Reimburse_Cashless_Encoded': 'mean'
}).reset_index()

# Flatten column names
policy_claims.columns = ['Nomor Polis', 'Total_Claims', 
                         'Total_Claim_Amount', 'Avg_Claim_Amount', 'Std_Claim_Amount', 'Min_Claim', 'Max_Claim',
                         'Total_Biaya_RS', 'Avg_Biaya_RS',
                         'Total_LOS', 'Avg_LOS',
                         'Inpatient_Ratio', 'Cashless_Ratio']

# Fill NaN values
policy_claims = policy_claims.fillna(0)

# Merge with policy data
df_polis_enhanced = df_polis.merge(policy_claims, on='Nomor Polis', how='left')
df_polis_enhanced = df_polis_enhanced.fillna(0)

# ============================================================
# 4. TIME SERIES AGGREGATION - MONTHLY LEVEL
# ============================================================
print("\nCreating monthly time series...")

# Aggregate claims by month
monthly_claims = df_klaim.groupby('Claim_Month').agg({
    'Claim ID': 'count',
    'Nominal Klaim Yang Disetujui': 'sum',
    'Nominal Biaya RS Yang Terjadi': 'sum',
    'Nomor Polis': 'nunique'
}).reset_index()

monthly_claims.columns = ['Month', 'Claim_Frequency', 'Total_Claim_Amount', 'Total_Biaya_RS', 'Unique_Policies']

# Calculate severity (average claim amount)
monthly_claims['Claim_Severity'] = monthly_claims['Total_Claim_Amount'] / monthly_claims['Claim_Frequency']
monthly_claims['Claim_Severity'] = monthly_claims['Claim_Severity'].fillna(0)

# Convert Period to datetime for easier handling
monthly_claims['Month_Date'] = monthly_claims['Month'].dt.to_timestamp()
monthly_claims['Year'] = monthly_claims['Month_Date'].dt.year
monthly_claims['Month_Num'] = monthly_claims['Month_Date'].dt.month

# Sort by date
monthly_claims = monthly_claims.sort_values('Month_Date').reset_index(drop=True)

print(f"Monthly data range: {monthly_claims['Month_Date'].min()} to {monthly_claims['Month_Date'].max()}")
print(f"Total months: {len(monthly_claims)}")

# ============================================================
# 5. ADDITIONAL FEATURES FOR TIME SERIES
# ============================================================
print("\nCreating time series features...")

# Create lag features
def create_lag_features(df, col, lags=[1, 2, 3, 6, 12]):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

# Create rolling statistics
def create_rolling_features(df, col, windows=[3, 6, 12]):
    for window in windows:
        df[f'{col}_rolling_mean_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'{col}_rolling_std_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).std()
    return df

# Apply feature engineering
for col in ['Claim_Frequency', 'Claim_Severity', 'Total_Claim_Amount']:
    monthly_claims = create_lag_features(monthly_claims, col)
    monthly_claims = create_rolling_features(monthly_claims, col)

# Add time-based features
monthly_claims['Month_Sin'] = np.sin(2 * np.pi * monthly_claims['Month_Num'] / 12)
monthly_claims['Month_Cos'] = np.cos(2 * np.pi * monthly_claims['Month_Num'] / 12)

# Add trend feature
monthly_claims['Trend'] = np.arange(len(monthly_claims))

# Year-over-year growth (if available)
monthly_claims['YoY_Frequency'] = monthly_claims['Claim_Frequency'] / monthly_claims['Claim_Frequency'].shift(12) - 1
monthly_claims['YoY_Severity'] = monthly_claims['Claim_Severity'] / monthly_claims['Claim_Severity'].shift(12) - 1

# Fill NaN values
monthly_claims = monthly_claims.fillna(method='bfill').fillna(method='ffill').fillna(0)

print(f"Features created. Shape: {monthly_claims.shape}")

# ============================================================
# 6. MODEL TRAINING
# ============================================================
print("\nTraining models...")

# Define feature columns (exclude target and date columns)
feature_cols = [col for col in monthly_claims.columns if col not in 
                ['Month', 'Month_Date', 'Claim_Frequency', 'Claim_Severity', 'Total_Claim_Amount', 
                 'Total_Biaya_RS', 'Unique_Policies']]

# Prepare training data
train_data = monthly_claims.dropna()
X = train_data[feature_cols]

# Train separate models for each target
models = {}
targets = ['Claim_Frequency', 'Claim_Severity', 'Total_Claim_Amount']

for target in targets:
    print(f"\nTraining model for {target}...")
    y = train_data[target]
    
    # Use Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    
    model.fit(X, y)
    models[target] = model
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top 5 important features for {target}:")
    print(importance.head())

# ============================================================
# 7. FORECASTING FOR AUG-DEC 2025
# ============================================================
print("\nForecasting for August - December 2025...")

# Create future months
future_months = pd.date_range(start='2025-08-01', end='2025-12-01', freq='MS')
forecast_results = {}

# Get the last row of data for initialization
last_row = monthly_claims.iloc[-1:].copy()

# Create forecast dataframe
forecast_df = pd.DataFrame()
forecast_df['Month_Date'] = future_months
forecast_df['Month_Num'] = forecast_df['Month_Date'].dt.month
forecast_df['Year'] = forecast_df['Month_Date'].dt.year
forecast_df['Trend'] = np.arange(len(monthly_claims), len(monthly_claims) + len(future_months))
forecast_df['Month_Sin'] = np.sin(2 * np.pi * forecast_df['Month_Num'] / 12)
forecast_df['Month_Cos'] = np.cos(2 * np.pi * forecast_df['Month_Num'] / 12)

# Iterative forecasting
for i, month in enumerate(future_months):
    # Create feature row for current month
    current_features = forecast_df.iloc[i:i+1].copy()
    
    # Add lag features from previous predictions or actual data
    if i == 0:
        # First month: use last known values
        for col in ['Claim_Frequency', 'Claim_Severity', 'Total_Claim_Amount']:
            current_features[f'{col}_lag_1'] = monthly_claims[col].iloc[-1]
            current_features[f'{col}_lag_2'] = monthly_claims[col].iloc[-2]
            current_features[f'{col}_lag_3'] = monthly_claims[col].iloc[-3]
            current_features[f'{col}_lag_6'] = monthly_claims[col].iloc[-6] if len(monthly_claims) >= 6 else monthly_claims[col].iloc[-1]
            current_features[f'{col}_lag_12'] = monthly_claims[col].iloc[-12] if len(monthly_claims) >= 12 else monthly_claims[col].iloc[-1]
            
            # Rolling features
            current_features[f'{col}_rolling_mean_3'] = monthly_claims[col].iloc[-3:].mean()
            current_features[f'{col}_rolling_mean_6'] = monthly_claims[col].iloc[-6:].mean() if len(monthly_claims) >= 6 else monthly_claims[col].iloc[-3:].mean()
            current_features[f'{col}_rolling_mean_12'] = monthly_claims[col].iloc[-12:].mean() if len(monthly_claims) >= 12 else monthly_claims[col].iloc[-6:].mean()
            
            current_features[f'{col}_rolling_std_3'] = monthly_claims[col].iloc[-3:].std()
            current_features[f'{col}_rolling_std_6'] = monthly_claims[col].iloc[-6:].std() if len(monthly_claims) >= 6 else monthly_claims[col].iloc[-3:].std()
            current_features[f'{col}_rolling_std_12'] = monthly_claims[col].iloc[-12:].std() if len(monthly_claims) >= 12 else monthly_claims[col].iloc[-6:].std()
    else:
        # Use previous predictions
        for col in ['Claim_Frequency', 'Claim_Severity', 'Total_Claim_Amount']:
            # Update lag features
            prev_preds = [forecast_results[col][j] for j in range(i)]
            
            current_features[f'{col}_lag_1'] = prev_preds[-1] if prev_preds else monthly_claims[col].iloc[-1]
            current_features[f'{col}_lag_2'] = prev_preds[-2] if len(prev_preds) >= 2 else monthly_claims[col].iloc[-2]
            current_features[f'{col}_lag_3'] = prev_preds[-3] if len(prev_preds) >= 3 else monthly_claims[col].iloc[-3]
            
            # Combine with historical data for longer lags
            hist_values = monthly_claims[col].tolist()
            all_values = hist_values + prev_preds
            
            current_features[f'{col}_lag_6'] = all_values[-6] if len(all_values) >= 6 else all_values[0]
            current_features[f'{col}_lag_12'] = all_values[-12] if len(all_values) >= 12 else all_values[0]
            
            # Rolling features
            current_features[f'{col}_rolling_mean_3'] = np.mean(all_values[-3:])
            current_features[f'{col}_rolling_mean_6'] = np.mean(all_values[-6:]) if len(all_values) >= 6 else np.mean(all_values)
            current_features[f'{col}_rolling_mean_12'] = np.mean(all_values[-12:]) if len(all_values) >= 12 else np.mean(all_values)
            
            current_features[f'{col}_rolling_std_3'] = np.std(all_values[-3:]) if len(all_values) >= 3 else 0
            current_features[f'{col}_rolling_std_6'] = np.std(all_values[-6:]) if len(all_values) >= 6 else np.std(all_values)
            current_features[f'{col}_rolling_std_12'] = np.std(all_values[-12:]) if len(all_values) >= 12 else np.std(all_values)
    
    # Add YoY features
    current_features['YoY_Frequency'] = 0
    current_features['YoY_Severity'] = 0
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in current_features.columns:
            current_features[col] = 0
    
    # Make predictions
    X_pred = current_features[feature_cols]
    
    for target in targets:
        pred = models[target].predict(X_pred)[0]
        # Ensure non-negative predictions
        pred = max(0, pred)
        
        if target not in forecast_results:
            forecast_results[target] = []
        forecast_results[target].append(pred)

# ============================================================
# 8. CREATE SUBMISSION FILE
# ============================================================
print("\nCreating submission file...")

submission_data = []

for i, month in enumerate(future_months):
    month_str = month.strftime('%Y_%m')
    
    submission_data.append({
        'id': f'{month_str}_Claim_Frequency',
        'value': round(forecast_results['Claim_Frequency'][i], 2)
    })
    submission_data.append({
        'id': f'{month_str}_Claim_Severity',
        'value': round(forecast_results['Claim_Severity'][i], 2)
    })
    submission_data.append({
        'id': f'{month_str}_Total_Claim',
        'value': round(forecast_results['Total_Claim_Amount'][i], 2)
    })

submission_df = pd.DataFrame(submission_data)

# Save submission
submission_df.to_csv('submission.csv', index=False)
print(f"\nSubmission saved to 'submission.csv'")
print(submission_df)

# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================
print("\n" + "="*60)
print("FORECAST SUMMARY")
print("="*60)

for i, month in enumerate(future_months):
    month_str = month.strftime('%Y-%m')
    print(f"\n{month_str}:")
    print(f"  Claim Frequency: {forecast_results['Claim_Frequency'][i]:,.2f}")
    print(f"  Claim Severity:  {forecast_results['Claim_Severity'][i]:,.2f}")
    print(f"  Total Claim:     {forecast_results['Total_Claim_Amount'][i]:,.2f}")

# Historical comparison
print("\n" + "="*60)
print("HISTORICAL COMPARISON (Last 6 months)")
print("="*60)
print(monthly_claims[['Month_Date', 'Claim_Frequency', 'Claim_Severity', 'Total_Claim_Amount']].tail(6))

print("\n" + "="*60)
print("Script completed successfully!")
print("="*60)
