# Retail Demand Forecasting with XGBoost

Also available as an interactive notebook on Kaggle: https://www.kaggle.com/code/tahsinbillah2k6/demand-forecasting

Predicting weekly units sold at the store-product level, using the Analytics Vidhya
JanataHack Demand Forecasting dataset (13,287 competition participants). Built as a
portfolio project to demonstrate applied machine learning workflow, not just model fitting.

**Final model:** 76.80% R², 14.31 MAE — a 27.7% improvement over a naive last-week-again
baseline — and 0.4247 RMSLE against the dataset's original competition scoring metric.

## What's actually in here

This isn't just "trained a model, got a score." Along the way I found and fixed two real
bugs in my own evaluation pipeline:
- A train/test leakage bug where I was unknowingly testing the model on data it had
  already trained on (fake R² of 72.32%, corrected to an honest 66.74%).
- A cross-validation fold-ordering bug in hyperparameter tuning, where `TimeSeriesSplit`
  was silently splitting by product identity instead of by time.

I also tested and rejected two ideas that seemed reasonable going in — one-hot encoding
for store/product IDs, and a 50/50 ensemble blend — and explained why each one
underperformed instead of just discarding them quietly. Full reasoning for all of this is
in the notebook itself, as markdown cells alongside the code.

**Honest limitation:** the added trend features improved aggregate performance but made
directional bias *worse* for my highest-volume, hardest-to-forecast stores — a reminder
that an average improvement can still hide a worse outcome for the segment that matters most.

## Dataset

[JanataHack: Demand Forecasting](https://www.kaggle.com/datasets/aswathrao/demand-forecasting) (Kaggle)

## How to run

Open `demand-forecasting.ipynb` on Kaggle, or download it and run locally — the first
cell installs everything it needs and fetches the dataset automatically either way.

## Setup

All required libraries are installed automatically by the first cell in the notebook — just open it and run everything top to bottom, no manual setup needed.

If you'd rather install the libraries yourself first, or just want to see what's used:

pip install -r requirements.txt