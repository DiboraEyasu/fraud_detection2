# fraud_country_stats.py

import pandas as pd


def get_country_fraud_stats(df: pd.DataFrame, target_col: str = "class") -> pd.DataFrame:
    """
    Compute fraud statistics per country.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least 'country' and the target_col.
    target_col : str
        Column name representing fraud indicator (0/1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - country
        - total_transactions
        - fraud_count
        - fraud_rate (0-1)
        Sorted descending by fraud_count.
    """
    stats = (
        df.groupby("country", as_index=False)
        .agg(
            total_transactions=("country", "size"),
            fraud_count=(target_col, "sum"),
            fraud_rate=(target_col, "mean"),
        )
        .sort_values("fraud_count", ascending=False)
    )
    return stats


def get_top_countries_by_fraud_count(
    country_stats: pd.DataFrame,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Return top N countries by fraud count.

    Parameters
    ----------
    country_stats : pd.DataFrame
        Output of get_country_fraud_stats().
    top_n : int
        Number of top countries to return.

    Returns
    -------
    pd.DataFrame
    """
    return country_stats.head(top_n)


def get_top_countries_by_fraud_rate(
    country_stats: pd.DataFrame,
    min_transactions: int = 100,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Return top N countries by fraud rate, filtered by minimum transaction count.

    Parameters
    ----------
    country_stats : pd.DataFrame
        Output of get_country_fraud_stats().
    min_transactions : int
        Minimum number of transactions to include a country.
    top_n : int
        Number of top countries to return.

    Returns
    -------
    pd.DataFrame
    """
    filtered = country_stats[
        country_stats["total_transactions"] >= min_transactions
    ]
    return filtered.sort_values("fraud_rate", ascending=False).head(top_n)