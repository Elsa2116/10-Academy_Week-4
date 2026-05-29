"""Feature engineering and proxy target creation for the credit risk project."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
LOGGER = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for transaction processing."""

    n_clusters: int = 3
    random_state: int = RANDOM_STATE


REQUIRED_COLUMNS = {
    "CustomerId",
    "Amount",
    "Value",
    "TransactionStartTime",
}


def validate_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise a helpful error if required columns are missing."""
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day, month, and year from TransactionStartTime."""
    validate_columns(df, {"TransactionStartTime"})
    out = df.copy()
    ts = pd.to_datetime(out["TransactionStartTime"], errors="coerce", utc=True)
    out["transaction_hour"] = ts.dt.hour
    out["transaction_day"] = ts.dt.day
    out["transaction_month"] = ts.dt.month
    out["transaction_year"] = ts.dt.year
    return out


def create_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level aggregate transaction features."""
    validate_columns(df, {"CustomerId", "Amount", "Value"})
    count_source = "TransactionId" if "TransactionId" in df.columns else "Amount"
    aggregates = (
        df.groupby("CustomerId")
        .agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            transaction_count=(count_source, "count"),
            std_amount=("Amount", "std"),
            total_value=("Value", "sum"),
            avg_value=("Value", "mean"),
        )
        .reset_index()
    )
    aggregates["std_amount"] = aggregates["std_amount"].fillna(0.0)
    return aggregates


def calculate_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Calculate Recency, Frequency, and Monetary metrics by customer."""
    validate_columns(df)
    work = df.copy()
    work["TransactionStartTime"] = pd.to_datetime(
        work["TransactionStartTime"], errors="coerce", utc=True
    )
    if snapshot_date is None:
        snapshot_date = work["TransactionStartTime"].max() + pd.Timedelta(days=1)
    snapshot_date = pd.Timestamp(snapshot_date)
    if snapshot_date.tzinfo is None:
        snapshot_date = snapshot_date.tz_localize("UTC")

    count_source = "TransactionId" if "TransactionId" in work.columns else "Amount"
    rfm = (
        work.groupby("CustomerId")
        .agg(
            last_transaction=("TransactionStartTime", "max"),
            frequency=(count_source, "count"),
            monetary=("Value", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (snapshot_date - rfm["last_transaction"]).dt.days
    return rfm.drop(columns=["last_transaction"])


def assign_high_risk_label(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Cluster RFM records and flag the least engaged segment as high risk."""
    required = {"CustomerId", "recency", "frequency", "monetary"}
    validate_columns(rfm, required)

    features = rfm[["recency", "frequency", "monetary"]].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled)

    labeled = rfm.copy()
    labeled["risk_cluster"] = clusters
    cluster_profile = labeled.groupby("risk_cluster")[["recency", "frequency", "monetary"]].mean()
    # High risk = older activity and weaker engagement. Recency is risk-positive;
    # frequency and monetary are risk-negative.
    risk_score = (
        cluster_profile["recency"].rank(ascending=True)
        + cluster_profile["frequency"].rank(ascending=False)
        + cluster_profile["monetary"].rank(ascending=False)
    )
    high_risk_cluster = int(risk_score.idxmax())
    labeled["is_high_risk"] = (labeled["risk_cluster"] == high_risk_cluster).astype(int)
    return labeled


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = "is_high_risk",
    bins: int = 5,
) -> tuple[pd.DataFrame, float]:
    """Calculate Weight of Evidence and Information Value for one feature.

    Numeric features are quantile-binned; categorical features are grouped by
    category. The returned table can be used for credit-scorecard diagnostics.
    """
    if feature not in df.columns or target not in df.columns:
        raise ValueError(f"Expected columns {feature!r} and {target!r}.")

    work = df[[feature, target]].copy()
    if pd.api.types.is_numeric_dtype(work[feature]) and work[feature].nunique(dropna=True) > bins:
        work["_bucket"] = pd.qcut(work[feature], q=bins, duplicates="drop")
    else:
        work["_bucket"] = work[feature].astype("object").fillna("Missing")

    grouped = work.groupby("_bucket", observed=False)[target].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "bad"})
    grouped["good"] = grouped["count"] - grouped["bad"]

    total_bad = max(grouped["bad"].sum(), 1)
    total_good = max(grouped["good"].sum(), 1)
    eps = 0.5
    grouped["bad_dist"] = (grouped["bad"] + eps) / (total_bad + eps * len(grouped))
    grouped["good_dist"] = (grouped["good"] + eps) / (total_good + eps * len(grouped))
    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
    grouped["iv"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]
    return grouped.reset_index().rename(columns={"_bucket": "bucket"}), float(grouped["iv"].sum())


def add_woe_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "is_high_risk",
    bins: int = 5,
) -> pd.DataFrame:
    """Append a WoE encoded diagnostic column for a feature."""
    woe_table, _ = calculate_woe_iv(df, feature=feature, target=target, bins=bins)
    out = df.copy()
    if pd.api.types.is_numeric_dtype(out[feature]) and out[feature].nunique(dropna=True) > bins:
        buckets = pd.qcut(out[feature], q=bins, duplicates="drop")
    else:
        buckets = out[feature].astype("object").fillna("Missing")
    mapping = dict(zip(woe_table["bucket"], woe_table["woe"]))
    out[f"{feature}_woe"] = pd.Series(buckets, index=out.index).map(mapping).astype(float)
    return out


class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that returns a model-ready DataFrame."""

    def __init__(self) -> None:
        self.feature_names_: list[str] = []
        self.preprocessor_: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        transformers = []
        if numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_cols,
                )
            )
        self.preprocessor_ = ColumnTransformer(transformers=transformers, remainder="drop")
        self.preprocessor_.fit(X)
        self.feature_names_ = self.preprocessor_.get_feature_names_out().tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor_ is None:
            raise RuntimeError("Transformer has not been fitted.")
        arr = self.preprocessor_.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_, index=X.index)


def build_feature_pipeline() -> Pipeline:
    """Return a single fitted-compatible Pipeline object for final preprocessing."""
    return Pipeline(steps=[("preprocessor", DataFramePreprocessor())])


def build_modeling_table(
    raw: pd.DataFrame,
    config: ProcessingConfig = ProcessingConfig(),
) -> pd.DataFrame:
    """Transform raw transactions into customer-level modeling table with target."""
    validate_columns(raw)
    enriched = add_datetime_features(raw)
    aggregates = create_customer_aggregates(enriched)
    rfm = calculate_rfm(enriched)
    labeled_rfm = assign_high_risk_label(
        rfm,
        n_clusters=config.n_clusters,
        random_state=config.random_state,
    )

    categorical_candidates = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]
    customer_cats = []
    for col in categorical_candidates:
        if col in enriched.columns:
            mode_values = (
                enriched.groupby("CustomerId")[col]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
                .rename(f"most_common_{col}")
            )
            customer_cats.append(mode_values)

    modeling = aggregates.merge(labeled_rfm, on="CustomerId", how="left")
    if customer_cats:
        cat_df = pd.concat(customer_cats, axis=1).reset_index()
        modeling = modeling.merge(cat_df, on="CustomerId", how="left")
    return modeling


def save_processed(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Load raw CSV, produce processed modeling table, and save it."""
    raw = pd.read_csv(input_path)
    processed = build_modeling_table(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    LOGGER.info("Saved processed data to %s with shape %s", output_path, processed.shape)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed credit-risk dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Raw transaction CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Processed CSV output path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    save_processed(args.input, args.output)


if __name__ == "__main__":
    main()
