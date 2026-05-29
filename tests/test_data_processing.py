import pandas as pd

from src.data_processing import (
    add_datetime_features,
    assign_high_risk_label,
    build_modeling_table,
    calculate_rfm,
    calculate_woe_iv,
    create_customer_aggregates,
)


def sample_transactions():
    return pd.DataFrame(
        {
            "TransactionId": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "CustomerId": ["C1", "C1", "C2", "C2", "C3", "C4"],
            "Amount": [100, 200, 50, 75, 10, 500],
            "Value": [100, 200, 50, 75, 10, 500],
            "TransactionStartTime": [
                "2024-01-01T10:00:00Z",
                "2024-01-02T11:00:00Z",
                "2024-01-03T12:00:00Z",
                "2024-01-04T13:00:00Z",
                "2023-12-01T08:00:00Z",
                "2024-01-04T09:00:00Z",
            ],
            "ProductCategory": ["airtime", "data", "airtime", "utility", "data", "airtime"],
            "ChannelId": ["web", "android", "web", "ios", "web", "android"],
        }
    )


def test_add_datetime_features_returns_expected_columns():
    df = add_datetime_features(sample_transactions())
    for col in ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]:
        assert col in df.columns
    assert df.loc[0, "transaction_hour"] == 10


def test_customer_aggregates_values_are_correct():
    agg = create_customer_aggregates(sample_transactions())
    c1 = agg.loc[agg["CustomerId"] == "C1"].iloc[0]
    assert c1["total_amount"] == 300
    assert c1["transaction_count"] == 2


def test_rfm_and_high_risk_label_are_created():
    rfm = calculate_rfm(sample_transactions(), snapshot_date=pd.Timestamp("2024-01-05", tz="UTC"))
    labeled = assign_high_risk_label(rfm, n_clusters=3, random_state=42)
    assert {"recency", "frequency", "monetary", "is_high_risk"}.issubset(labeled.columns)
    assert set(labeled["is_high_risk"].unique()).issubset({0, 1})


def test_build_modeling_table_contains_target():
    table = build_modeling_table(sample_transactions())
    assert "is_high_risk" in table.columns
    assert table["CustomerId"].nunique() == 4


def test_calculate_woe_iv_returns_diagnostics():
    table = build_modeling_table(sample_transactions())
    woe_table, iv = calculate_woe_iv(table, "transaction_count")
    assert {"bucket", "woe", "iv"}.issubset(woe_table.columns)
    assert iv >= 0
