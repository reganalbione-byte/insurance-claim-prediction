# Health Insurance Claim Prediction

Regression model to forecast health insurance claim amounts based on policyholder data. Built for the **Machine Learning Competition** hosted by MCF ITB in collaboration with AXA Financial Indonesia.

## Problem

Given historical policyholder and claims data (`Data_Polis.csv` and `Data_Klaim.csv`), predict future insurance claim amounts. The goal is to help insurers better estimate reserves and identify high-risk policyholders.

## Approach

1. **Data Exploration** -- Analyzed claim distributions, policyholder demographics, and temporal patterns across the two datasets. Identified skewed claim amounts and seasonal trends.

2. **Feature Engineering** -- Created interaction terms between policyholder attributes, binned continuous variables for better model fit, and encoded categorical variables. Merged policy and claims data on shared keys.

3. **Modeling** -- Trained and evaluated multiple approaches:
   - **Ridge Regression** -- Primary model, chosen for stability with correlated features
   - Experimented with ensemble methods, exponential smoothing, and frequency-severity decomposition during validation

4. **Evaluation** -- Assessed model performance using RMSE and MAE metrics against holdout validation sets.

## Tech Stack

- Python
- Scikit-learn (Ridge, preprocessing, metrics)
- Pandas, NumPy
- Matplotlib, Seaborn (EDA visualizations)

## Project Structure

```
claim_prediction.ipynb   # Full EDA + modeling pipeline
submission.csv           # Final competition submission
README.md
```

## Results

Submitted predictions for the MCF ITB x AXA Financial competition leaderboard. The Ridge Regression model balanced bias-variance tradeoff well for this relatively small, tabular dataset.

## What I Learned

- Insurance claims data is heavily right-skewed -- log transforms and robust scaling were essential during preprocessing.
- Frequency-severity decomposition (predicting claim count and claim amount separately, then combining) is a standard actuarial approach worth exploring for insurance prediction problems.
- Feature interactions between policyholder demographics and policy type carried more predictive signal than any single feature alone.
