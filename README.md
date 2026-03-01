# Employee Turnover Prediction

## Problem
Predict employee attrition using supervised machine learning and identify the structural drivers behind workforce churn. The goal is to move from reactive hiring to proactive retention by modeling turnover risk before critical inflection points (e.g., Year-4 tenure spike).

## Dataset
- Source: Internal HR survey dataset (Salifort Motors, a fictional company)
- Rows: ~11,000 employees
- Features: 10+ core variables (satisfaction, evaluation score, number of projects, monthly hours, tenure, salary, department, promotion history)
- Engineered Features: burnout_score and workload intensity indicators

## Tools
- Python  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Scikit-learn  
- Statsmodels (VIF & statistical validation)  
- XGBoost  

## Results
- Logistic Regression (Baseline)
  - Interpretable benchmark model
  - Identified salary and satisfaction as major linear predictors

- XGBoost (Final Model)
  - Accuracy: 99%
  - ROC–AUC: 0.983
  - F1-Score: 0.96
  - Captured non-linear patterns including:
    - Year-4 tenure spike
    - U-shaped workload risk
    - Burnout-driven attrition dynamics

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt