# Interim Report: Credit Risk Probability Model for Alternative Data

## 1. Project understanding

Bati Bank wants to support a buy-now-pay-later product using behavioral transaction data from an eCommerce partner. The objective is to transform raw transaction records into a model that can estimate customer risk probability in real time.

Because the dataset does not contain an observed default label, the project uses a proxy target derived from customer behavior. The central assumption is that low engagement, low transaction frequency, and low monetary activity may indicate higher credit risk. This proxy is not ground truth and must be clearly documented for risk governance.

## 2. Basel II implications

Basel II expectations push the project toward reproducible, explainable, and well-documented modeling. The model development workflow therefore includes:

- clear data lineage from raw transactions to model-ready features;
- documented proxy target design;
- reproducible random states for clustering, splitting, and model training;
- interpretable benchmark models such as Logistic Regression;
- experiment tracking with MLflow;
- unit tests and CI/CD checks;
- model limitations and monitoring recommendations.

## 3. EDA plan and required outputs

The EDA notebook in `notebooks/eda.ipynb` covers:

- dataset shape, columns, and data types;
- summary statistics;
- numerical distributions;
- categorical distributions;
- missing values;
- correlation analysis;
- outlier detection using box plots;
- top 3-5 insights in a final markdown cell.

## 4. Expected EDA insights to validate after running data

After loading the real Xente data, the following should be checked and updated with actual values:

1. Transaction amount and value distributions are likely skewed and may contain extreme outliers.
2. Product category and channel usage may be concentrated in a small number of categories.
3. FraudResult is likely highly imbalanced and should not be used as a default label.
4. Customer transaction counts are likely uneven, supporting RFM-based customer aggregation.
5. Missing values and duplicated identifiers should be reviewed before production feature engineering.

## 5. Interim deliverables checklist

- [x] Standard repository structure.
- [x] README business understanding section.
- [x] EDA notebook template.
- [x] Interim report.
- [ ] Run notebook on actual raw CSV.
- [ ] Replace expected insights with actual findings.
- [ ] Add screenshots or exported plots if required by instructor.

## 6. Important reminder

The task brief does not specify a page limit for the interim report. Keep the report concise, but make sure it covers project understanding and EDA findings.
