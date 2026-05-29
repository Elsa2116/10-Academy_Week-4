# From Transactions to Trust: Building an Alternative Credit Risk Model for Bati Bank

## Executive Summary

Bati Bank wants to launch a buy-now-pay-later product with an eCommerce partner. The core challenge is that the available Xente transaction data describes customer behavior, but it does not contain a true credit default outcome. This project therefore builds a complete prototype for alternative-data credit scoring: data exploration, feature engineering, proxy target construction, model training with MLflow tracking, FastAPI deployment, Docker packaging, and CI/CD testing.

The model output should be treated as a behavioral risk signal, not a fully validated regulatory probability of default. Before production use for automated credit decisions, Bati Bank should validate the model against real repayment outcomes.

## Business Problem

Traditional credit scoring uses historical repayment and default behavior. In this project, Bati Bank has transaction-level eCommerce data instead. The product goal is to estimate whether a customer is likely to be high risk before issuing BNPL credit, then use that signal to support approval, limit, amount, and duration decisions.

Basel II expectations matter because credit models must be measurable, explainable, documented, monitored, and challengeable. For that reason, this implementation uses reproducible random states, documented proxy logic, interpretable benchmark models, tracked experiments, unit tests, and a deployment path that can be audited.

## Proxy Variable Justification

The raw data has no direct `default` label. A supervised credit risk model still needs a target, so this project creates a proxy target using Recency, Frequency, and Monetary value behavior.

The modeling assumption is:

- customers with recent, frequent, and higher-value activity are more engaged and lower behavioral risk;
- customers with stale, infrequent, and low-value activity are less engaged and higher behavioral risk.

This produces a binary target named `is_high_risk`. It is useful for prototyping and ranking customers, but it is not ground truth. The main business risks are proxy bias, false rejection of good customers, false approval of risky customers, seasonality, and missing information about income or repayment capacity.

## RFM Clustering Methodology

The processing workflow in `src/data_processing.py` builds customer-level records from raw transactions.

1. Convert `TransactionStartTime` to datetime features.
2. Aggregate customer behavior: total amount, average amount, transaction count, standard deviation, total value, and average value.
3. Calculate RFM metrics:
   - `recency`: days since the customer's last transaction from a fixed snapshot date;
   - `frequency`: number of transactions;
   - `monetary`: total transaction value.
4. Scale the RFM features with `StandardScaler`.
5. Cluster customers into three groups using KMeans with `random_state=42`.
6. Identify the least engaged cluster using high recency, low frequency, and low monetary value.
7. Assign `is_high_risk=1` to the least engaged cluster and `0` to all other customers.

The processed dataset is written to `data/processed/processed_credit_risk.csv` when the raw Xente CSV is available.

## Feature Engineering

The model-ready table includes:

- customer aggregates: `total_amount`, `avg_amount`, `transaction_count`, `std_amount`, `total_value`, `avg_value`;
- RFM features: `recency`, `frequency`, `monetary`;
- categorical summaries such as most common provider, product, product category, channel, and pricing strategy;
- missing-value handling, one-hot encoding, and scaling through an sklearn `Pipeline`;
- WoE and IV helper functions for credit-scorecard diagnostics.

This keeps exploratory analysis separate from production transformation logic.

## Model Training and Tracking

The training workflow in `src/train.py` compares three model families:

| Model | Purpose |
| --- | --- |
| Logistic Regression | Interpretable Basel-friendly benchmark |
| Random Forest | Nonlinear ensemble candidate |
| Gradient Boosting | Higher-performance boosting candidate |

Each model is tuned with `GridSearchCV`, evaluated on a held-out test set, and logged to MLflow with parameters, metrics, and artifacts. The best model is selected by ROC-AUC when both classes are present, saved as `data/processed/best_model.joblib`, and registered as `CreditRiskProbabilityModel`.

## Model Comparison Results

Run the command below after placing the real Xente CSV in `data/raw/`:

```bash
python -m src.train --input data/raw/training.csv --experiment-name credit-risk-model
```

Then update this table from MLflow and `data/processed/model_metrics.json`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | Most interpretable benchmark |
| Random Forest | TBD | TBD | TBD | TBD | TBD | Captures nonlinear behavior |
| Gradient Boosting | TBD | TBD | TBD | TBD | TBD | Strong performance candidate |

## API Demonstration

The FastAPI service loads `data/processed/best_model.joblib` when available and falls back to the MLflow registered model URI.

Start locally:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Sample request:

```json
{
  "features": {
    "total_amount": 5000,
    "avg_amount": 250,
    "transaction_count": 20,
    "std_amount": 100,
    "total_value": 5000,
    "avg_value": 250,
    "recency": 7,
    "frequency": 20,
    "monetary": 5000
  }
}
```

Sample response:

```json
{
  "risk_probability": 0.31,
  "is_high_risk": 0,
  "credit_score": 680
}
```

The credit score maps risk probability to a 300-850 range where lower risk receives a higher score.

## Deployment and CI/CD

The repository includes:

- `Dockerfile` for containerizing the API;
- `docker-compose.yml` for local service startup;
- `.github/workflows/ci.yml` for linting and unit tests on pushes and pull requests to `main`;
- `tests/test_data_processing.py` for data-processing and WoE/IV tests.

Run the containerized service:

```bash
docker compose up --build
```

Run local checks:

```bash
pytest -q
flake8 src tests --max-line-length=100
```

## Screenshots to Add Before Submission

Add screenshots in `reports/figures/` for:

1. MLflow experiment runs and metrics.
2. MLflow registered model page for `CreditRiskProbabilityModel`.
3. GitHub Actions passing status.
4. Docker container running successfully.
5. FastAPI `/docs` page or successful `/predict` request.

## Limitations

The proxy target estimates behavioral disengagement risk, not verified credit default. It may misclassify customers whose transaction behavior does not reflect repayment ability. The data may also contain channel, product, seasonality, or access bias. A bank production deployment would require real repayment outcomes, ongoing drift monitoring, fairness analysis, challenger models, and governance sign-off.

## Recommendations

1. Use this model as a decision-support signal during prototyping.
2. Collect actual repayment, delinquency, and default outcomes as soon as BNPL loans are issued.
3. Recalibrate or replace the proxy target when true loan performance data is available.
4. Keep Logistic Regression/WoE as an interpretable benchmark even if an ensemble model wins on ROC-AUC.
5. Monitor approval rates, default rates, score drift, feature drift, and customer-segment impacts.

## Conclusion

This project delivers a complete alternative-data credit risk prototype for Bati Bank. It is ready to run once the Xente transaction CSV is added to `data/raw/`, and its structure supports the next stage: validating the proxy-driven model against real credit outcomes.
