# FraudShield: Credit Card Fraud Detection

![alt text](<Screenshot 2026-11-07 171220.png>)

FraudShield-CreditCard - https://fraudshield-creditcard.streamlit.app/

A fraud detection system evaluated the way production works: trained on the past,
tested on the future, with the alert threshold chosen from money, not from 0.5.

## Results (time-based test set, 56,962 transactions, 75 frauds)

| Model | PR-AUC | Recall at 90% precision |
|---|---|---|
| Logistic regression | 0.7484 | 69.33% |
| Random forest | 0.8207 | 76.00% |
| **XGBoost (deployed)** | **0.7947** | **76.00%** |
| Isolation Forest (unsupervised) | 0.0460 | 0.00% |

At the chosen threshold: **63** alerts, **57** frauds caught, **18** missed, net savings **$4,776** (assuming a review cost of $5 per alert).

## What is different from a typical Kaggle fraud notebook

- Time-based train/validation/test split, no random shuffling
- SMOTE only inside a pipeline, compared against class weights and no resampling
- PR-AUC and recall at fixed precision instead of accuracy
- Probability calibration and a cost-based threshold
- SHAP explanations, a cost simulator and an analyst review queue

## Run it

    pip install -r requirements.txt
    python -m src.train          # needs data/creditcard.csv from Kaggle (ULB dataset)
    streamlit run app/dashboard.py

## Limitations

- Features V1 to V28 are anonymised PCA components, so feature engineering is limited.
- Only about 75 frauds in the test set, so metrics are noisy.
- The data covers two days in 2013; real fraud patterns drift over time.
- The review cost is an assumption.
- The hour feature is hours since the start of the data, not a real clock time.

## Next steps

FastAPI scoring service, Docker, CI, drift monitoring.