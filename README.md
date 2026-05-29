# Credit Risk Probability Model for Alternative Data

End-to-end implementation for building, tracking, deploying, and testing a credit risk probability model for Bati Bank's buy-now-pay-later use case.

## Project Structure

```text
credit-risk-model/
|-- .github/workflows/ci.yml
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/eda.ipynb
|-- reports/
|   |-- interim_report.md
|   `-- final_report.md
|-- src/
|   |-- data_processing.py
|   |-- train.py
|   |-- predict.py
|   `-- api/
|       |-- main.py
|       `-- pydantic_models.py
|-- tests/test_data_processing.py
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Credit Scoring Business Understanding

### Basel II and model interpretability

The Basel II Accord emphasizes risk measurement, governance, documentation, and validation. In this project, that means the model cannot be treated as a black-box prediction engine only. It must produce defensible probability-of-risk estimates, use reproducible transformations, preserve training evidence, and be monitored after deployment. Because a credit decision may affect customer access to credit, model documentation should explain the data, feature engineering logic, proxy target definition, evaluation metrics, limitations, and operational controls.

Interpretable modeling choices are especially important because the available target is not a true observed default outcome. Logistic Regression with documented variables, Weight of Evidence transformations, and Information Value analysis can help risk teams understand directional effects. More complex models can still be used, but they need extra explanation, experiment tracking, validation, and approval evidence.

### Why a proxy variable is necessary

The raw eCommerce transaction data does not include a direct loan default label. Without a target variable, supervised learning cannot directly learn default behavior. Therefore, this project creates a proxy target from customer behavior using Recency, Frequency, and Monetary value patterns. Customers with low engagement, low transaction frequency, and low monetary activity are treated as higher-risk proxies.

This proxy is useful for experimentation, but it introduces business risk. A disengaged customer is not necessarily a defaulter, and an active customer is not necessarily low risk. Proxy labels may encode behavioral bias, seasonality, channel effects, or product-access differences. For that reason, the model output should be framed as a behavioral risk signal, not a fully validated regulatory default model. A production bank deployment would require validation against real repayment/default outcomes when available.

### Trade-offs: interpretable vs high-performance models

A simple model such as Logistic Regression with WoE features is easier to explain, audit, document, and challenge. It supports Basel-style governance because feature effects can be reviewed by risk, compliance, and business teams. The trade-off is that it may underfit nonlinear relationships and interactions in transaction behavior.

A higher-performance model such as Random Forest or Gradient Boosting may capture nonlinear patterns and improve ROC-AUC, recall, or F1 score. The trade-off is reduced transparency, more complex validation, and higher monitoring burden. In a regulated credit context, the final model should balance performance with explainability. A strong workflow is to compare both families, select the best model using tracked metrics, and document why the chosen model is acceptable for the business and regulatory context.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Xente dataset from Kaggle and place the transaction CSV in `data/raw/`.

Expected filename examples:

```text
data/raw/training.csv
data/raw/xente_train.csv
data/raw/data.csv
```

## Run EDA

Open and run:

```bash
jupyter notebook notebooks/eda.ipynb
```

The notebook creates summary tables and visualizations for numerical distributions, categorical distributions, correlations, missing values, and outliers. Update the final markdown cell with the actual top 3-5 findings after the raw CSV is available.

## Build processed dataset

```bash
python -m src.data_processing --input data/raw/training.csv --output data/processed/processed_credit_risk.csv
```

## Train models with MLflow

```bash
python -m src.train --input data/raw/training.csv --experiment-name credit-risk-model
mlflow ui
```

## Run API locally

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"total_amount":5000,"avg_amount":250,"transaction_count":20,"std_amount":100,"recency":7,"frequency":20,"monetary":5000}}'
```

## Run with Docker

```bash
docker compose up --build
```

## Test and lint

```bash
pytest -q
flake8 src tests --max-line-length=100
```

## Submission Checklist

Interim submission:

- GitHub repository link, main branch.
- Task 1 README business understanding.
- Task 2 EDA notebook with top 3-5 insights.
- `reports/interim_report.md`.

Final submission:

- GitHub repository link, main branch.
- Complete code for Tasks 1-6.
- MLflow experiments and best model registration evidence.
- FastAPI prediction endpoint.
- Docker and docker-compose files.
- GitHub Actions CI/CD.
- `reports/final_report.md` in Medium-style format.
- Screenshots of MLflow tracking, CI/CD passing, and Docker/API running.

## Branch Workflow

```bash
git checkout -b task-1
git add . && git commit -m "Initialize project and business understanding"
git checkout main && git merge task-1

git checkout -b task-2
git add notebooks reports && git commit -m "Add EDA notebook and interim report"
git checkout main && git merge task-2

git checkout -b task-3
git add src/data_processing.py tests && git commit -m "Add feature engineering pipeline"
git checkout main && git merge task-3

git checkout -b task-4
git add src/data_processing.py && git commit -m "Add RFM proxy target engineering"
git checkout main && git merge task-4

git checkout -b task-5
git add src/train.py src/predict.py tests && git commit -m "Add model training and MLflow tracking"
git checkout main && git merge task-5

git checkout -b task-6
git add src/api Dockerfile docker-compose.yml .github/workflows/ci.yml
git commit -m "Add FastAPI deployment and CI"
git checkout main && git merge task-6
```
