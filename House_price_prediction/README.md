# House Price Prediction — Regression Model + Dashboard

## Files
- `train_model.py` — generates a realistic housing dataset, trains 4-5 regression
  models (Linear, Ridge, Lasso, Random Forest, XGBoost), picks the best by RMSE,
  and saves it along with metrics.
- `dashboard.py` — interactive Streamlit dashboard: data exploration, model
  comparison, and a live price-prediction form.

## Setup
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib xgboost plotly
```

## Run

1. Train the model first (creates `best_model.pkl`, `housing_data.csv`, etc.):
```bash
python train_model.py
```

2. Launch the dashboard:
```bash
streamlit run dashboard.py
```
Then open the URL Streamlit prints (usually http://localhost:8501).

## Notes
- The dataset is synthetically generated locally (no internet/download needed),
  modeled after real housing datasets like Ames Housing / California Housing —
  swap in your own CSV in `train_model.py`'s `load_data()` if you want real data.
- The dashboard auto-picks whichever model had the lowest RMSE during training.
