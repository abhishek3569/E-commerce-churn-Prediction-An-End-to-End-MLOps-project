import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


RANDOM_SEED = 42
DEFAULT_N_CUSTOMERS = 100_000


def _generate_demographics(n: int, rng: np.random.Generator) -> pd.DataFrame:
    customer_id = np.arange(1, n + 1)

    # Account age: between 30 and 2000 days, skewed toward newer customers
    account_age_days = rng.integers(30, 2000, size=n)

    customer_segment = rng.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        size=n,
        p=[0.5, 0.3, 0.15, 0.05],
    )

    signup_channel = rng.choice(
        ["organic", "paid_social", "email", "referral", "direct"],
        size=n,
        p=[0.4, 0.15, 0.2, 0.15, 0.1],
    )

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "account_age_days": account_age_days,
            "customer_segment": customer_segment,
            "signup_channel": signup_channel,
        }
    )


def _generate_transactions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    # Total purchases with many low-activity customers
    total_purchases = rng.poisson(lam=5, size=n)
    total_purchases = np.clip(total_purchases, 0, 100)

    # Average order value between 10 and 500, log-normal for skew
    aov = np.exp(rng.normal(loc=3.5, scale=0.6, size=n))
    aov = np.clip(aov, 10, 500)

    total_revenue = total_purchases * aov
    total_revenue = np.clip(total_revenue, 0, 50_000)

    # Days since last purchase, more recent for heavy buyers
    base_days = rng.exponential(scale=60, size=n)
    days_since_last_purchase = np.clip(base_days, 0, 365).astype(int)

    # Purchase frequency (purchases per month)
    account_age_months = np.maximum(1, rng.normal(loc=12, scale=6, size=n))
    purchase_frequency = total_purchases / account_age_months

    return pd.DataFrame(
        {
            "total_purchases": total_purchases,
            "total_revenue": total_revenue,
            "avg_order_value": aov,
            "days_since_last_purchase": days_since_last_purchase,
            "purchase_frequency": purchase_frequency,
        }
    )


def _generate_engagement(n: int, rng: np.random.Generator) -> pd.DataFrame:
    website_visits_last_30days = rng.poisson(lam=5, size=n)
    website_visits_last_30days = np.clip(website_visits_last_30days, 0, 100)

    # Email open & click rates using Beta distributions
    email_open_rate = rng.beta(a=2, b=5, size=n)
    email_click_rate = email_open_rate * rng.beta(a=2, b=8, size=n)

    app_usage_minutes_last_30days = rng.gamma(shape=2.0, scale=60.0, size=n)
    app_usage_minutes_last_30days = np.clip(app_usage_minutes_last_30days, 0, 1000)

    customer_service_contacts = rng.poisson(lam=0.5, size=n)
    customer_service_contacts = np.clip(customer_service_contacts, 0, 20)

    return pd.DataFrame(
        {
            "website_visits_last_30days": website_visits_last_30days,
            "email_open_rate": email_open_rate,
            "email_click_rate": email_click_rate,
            "app_usage_minutes_last_30days": app_usage_minutes_last_30days,
            "customer_service_contacts": customer_service_contacts,
        }
    )


def _generate_product_behavior(n: int, rng: np.random.Generator) -> pd.DataFrame:
    favorite_category = rng.choice(
        ["Electronics", "Fashion", "Home", "Sports", "Beauty"],
        size=n,
        p=[0.25, 0.25, 0.2, 0.15, 0.15],
    )

    number_of_categories_purchased = rng.integers(1, 11, size=n)

    returns_count = rng.poisson(lam=1.0, size=n)
    returns_count = np.clip(returns_count, 0, 20)

    # Return rate linked to returns_count and total_purchases (approximate)
    raw_return_rate = rng.beta(a=1.5, b=8, size=n)
    return_rate = np.clip(raw_return_rate + 0.02 * (returns_count > 3), 0.0, 1.0)

    return pd.DataFrame(
        {
            "favorite_category": favorite_category,
            "number_of_categories_purchased": number_of_categories_purchased,
            "returns_count": returns_count,
            "return_rate": return_rate,
        }
    )


def _generate_loyalty(n: int, rng: np.random.Generator) -> pd.DataFrame:
    loyalty_points_balance = rng.gamma(shape=2.0, scale=1000.0, size=n)
    loyalty_points_balance = np.clip(loyalty_points_balance, 0, 10_000)

    discount_usage_rate = rng.beta(a=2, b=5, size=n)

    referrals_made = rng.poisson(lam=0.2, size=n)
    referrals_made = np.clip(referrals_made, 0, 10)

    return pd.DataFrame(
        {
            "loyalty_points_balance": loyalty_points_balance,
            "discount_usage_rate": discount_usage_rate,
            "referrals_made": referrals_made,
        }
    )


def _compute_churn_probability(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Compute churn probability using a logistic-like function with business rules.
    """
    # Base log-odds
    log_odds = np.full(len(df), -0.5)  # baseline churn ~0.38

    # High churn risk factors
    log_odds += 1.2 * (df["days_since_last_purchase"] > 180).astype(float)
    log_odds += 0.8 * (df["website_visits_last_30days"] < 2).astype(float)
    log_odds += 0.9 * (df["email_open_rate"] < 0.1).astype(float)
    log_odds += 0.7 * (df["customer_service_contacts"] > 5).astype(float)
    log_odds += 0.8 * (df["return_rate"] > 0.3).astype(float)

    # Low churn protective factors
    log_odds -= 0.8 * df["customer_segment"].isin(["Gold", "Platinum"]).astype(float)
    log_odds -= 0.7 * (df["purchase_frequency"] > 2).astype(float)
    log_odds -= 0.6 * (df["loyalty_points_balance"] > 5000).astype(float)
    log_odds -= 0.5 * (df["app_usage_minutes_last_30days"] > 100).astype(float)

    # Add noise to avoid perfect separation
    log_odds += rng.normal(loc=0.0, scale=0.5, size=len(df))

    # Convert to probability
    probs = 1 / (1 + np.exp(-log_odds))
    return probs


def _generate_churn_label(df: pd.DataFrame, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    probs = _compute_churn_probability(df, rng)
    churned = rng.binomial(1, probs, size=len(df))
    return churned, probs


def _generate_dataset(n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    demo = _generate_demographics(n, rng)
    tx = _generate_transactions(n, rng)
    eng = _generate_engagement(n, rng)
    prod = _generate_product_behavior(n, rng)
    loy = _generate_loyalty(n, rng)

    df = pd.concat([demo, tx, eng, prod, loy], axis=1)

    churned, churn_probability = _generate_churn_label(df, rng)
    df["churn_probability"] = churn_probability
    df["churned"] = churned

    # Add reference date and generation timestamp
    now = datetime.utcnow()
    df["data_generation_timestamp"] = now.isoformat()

    # Approximate last_purchase_date using days_since_last_purchase
    df["last_purchase_date"] = (
        now - pd.to_timedelta(df["days_since_last_purchase"], unit="D")
    ).dt.date

    return df


def _data_quality_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("E-commerce Churn Synthetic Data Quality Report")
    lines.append(f"Generated at: {datetime.utcnow().isoformat()} UTC")
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    lines.append("")

    # Basic stats for numeric columns
    num_df = df.select_dtypes(include=[np.number])
    lines.append("Numeric Feature Summary:")
    lines.append(num_df.describe().to_string())
    lines.append("")

    # Churn distribution
    if "churned" in df.columns:
        churn_counts = df["churned"].value_counts(normalize=True).rename("fraction")
        lines.append("Churn Distribution (fraction):")
        lines.append(churn_counts.to_string())
        lines.append("")

    # Missing values
    missing = df.isna().mean()
    lines.append("Missing Value Fractions:")
    lines.append(missing[missing > 0].to_string() if (missing > 0).any() else "No missing values.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce churn dataset.")
    parser.add_argument(
        "--n_customers",
        type=int,
        default=DEFAULT_N_CUSTOMERS,
        help="Number of customer records to generate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save the generated CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    df = _generate_dataset(args.n_customers, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_suffix = datetime.utcnow().strftime("%Y%m%d")
    csv_path = output_dir / f"customers_{date_suffix}.csv"

    df.to_csv(csv_path, index=False)

    # Data quality report
    report = _data_quality_report(df)
    report_path = output_dir / f"data_quality_report_{date_suffix}.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved dataset to: {csv_path}")
    print(f"Saved data quality report to: {report_path}")


if __name__ == "__main__":
    main()

