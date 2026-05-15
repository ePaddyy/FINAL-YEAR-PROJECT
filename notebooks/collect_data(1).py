"""
Data Collection Script — Ghana Fuel Price Forecasting Project
Collects: Brent Crude Oil, GHS/USD Exchange Rate, CPI (Inflation)
Sources:  EIA API, Frankfurter API, World Bank API
Authors:  [Your Group Names]
Date:     2026
"""

import requests
import pandas as pd
from datetime import datetime
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
START_DATE = "2013-01-01"   # Change to match your study period
END_DATE   = "2023-12-31"   # Change to match your study period
EIA_API_KEY = "YOUR_EIA_API_KEY_HERE"  # Register free at https://www.eia.gov/opendata/


# ─────────────────────────────────────────────
# 1. BRENT CRUDE OIL PRICES — EIA API
# ─────────────────────────────────────────────
def get_brent_crude(api_key, start, end):
    print("Fetching Brent Crude Oil prices from EIA...")
    url = (
        f"https://api.eia.gov/v2/petroleum/pri/spt/data/"
        f"?api_key={api_key}"
        f"&frequency=monthly"
        f"&data[0]=value"
        f"&facets[series][]=RBRTE"
        f"&start={start[:7]}"
        f"&end={end[:7]}"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=asc"
        f"&offset=0&length=5000"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print(f"  ERROR: {response.status_code} - {response.text}")
        return None

    data = response.json()
    records = data.get("response", {}).get("data", [])
    if not records:
        print("  No data returned. Check your API key or date range.")
        return None

    df = pd.DataFrame(records)[["period", "value"]]
    df.columns = ["date", "brent_crude_usd"]
    df["date"] = pd.to_datetime(df["date"])
    df["brent_crude_usd"] = pd.to_numeric(df["brent_crude_usd"], errors="coerce")
    print(f"  Retrieved {len(df)} monthly observations.")
    return df


# ─────────────────────────────────────────────
# 2. GHS/USD EXCHANGE RATE — Frankfurter API
# ─────────────────────────────────────────────
def get_exchange_rate(start, end):
    print("Fetching GHS/USD exchange rate from Frankfurter...")
    records = []
    # Frankfurter returns daily — we will resample to monthly
    url = f"https://api.frankfurter.app/{start}..{end}?from=USD&to=GHS"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"  ERROR: {response.status_code} - {response.text}")
        return None

    data = response.json()
    rates = data.get("rates", {})

    for date_str, currency_dict in rates.items():
        ghs_rate = currency_dict.get("GHS")
        if ghs_rate:
            records.append({"date": date_str, "exchange_rate_ghs_usd": ghs_rate})

    if not records:
        print("  No data returned.")
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Resample to monthly — take last observation of each month
    df_monthly = df.resample("MS").last().reset_index()
    print(f"  Retrieved {len(df_monthly)} monthly observations.")
    return df_monthly


# ─────────────────────────────────────────────
# 3. CPI / INFLATION — World Bank API
# ─────────────────────────────────────────────
def get_cpi_worldbank(start, end):
    print("Fetching CPI data from World Bank...")
    start_year = int(start[:4])
    end_year   = int(end[:4])

    url = (
        f"https://api.worldbank.org/v2/country/GH/indicator/FP.CPI.TOTL"
        f"?date={start_year}:{end_year}&format=json&per_page=100"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print(f"  ERROR: {response.status_code}")
        return None

    data = response.json()
    if len(data) < 2 or not data[1]:
        print("  No data returned.")
        return None

    records = []
    for entry in data[1]:
        if entry.get("value") is not None:
            records.append({
                "date": pd.to_datetime(f"{entry['date']}-01-01"),
                "cpi": entry["value"]
            })

    if not records:
        return None

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    # World Bank CPI is annual — interpolate to monthly
    df = df.set_index("date")
    df_monthly = df.resample("MS").interpolate(method="linear").reset_index()
    df_monthly = df_monthly[
        (df_monthly["date"] >= START_DATE) &
        (df_monthly["date"] <= END_DATE)
    ]
    print(f"  Retrieved {len(df_monthly)} monthly observations (interpolated from annual).")
    print("  NOTE: CPI is annual from World Bank — interpolated to monthly.")
    print("  Consider replacing with GSS monthly CPI reports for higher accuracy.")
    return df_monthly


# ─────────────────────────────────────────────
# 4. MERGE ALL DATASETS
# ─────────────────────────────────────────────
def merge_datasets(brent_df, fx_df, cpi_df):
    print("\nMerging all datasets...")

    # Standardise date column to month start
    for df in [brent_df, fx_df, cpi_df]:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    merged = brent_df.merge(fx_df, on="date", how="outer")
    merged = merged.merge(cpi_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Filter to study period
    merged = merged[
        (merged["date"] >= START_DATE) &
        (merged["date"] <= END_DATE)
    ]

    print(f"  Final dataset: {len(merged)} rows x {len(merged.columns)} columns")
    print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"  Missing values:\n{merged.isnull().sum()}")
    return merged


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Ghana Fuel Price Project — Data Collection")
    print("=" * 55)

    brent_df = get_brent_crude(EIA_API_KEY, START_DATE, END_DATE)
    time.sleep(1)

    fx_df = get_exchange_rate(START_DATE, END_DATE)
    time.sleep(1)

    cpi_df = get_cpi_worldbank(START_DATE, END_DATE)

    if brent_df is not None and fx_df is not None and cpi_df is not None:
        final_df = merge_datasets(brent_df, fx_df, cpi_df)
        output_path = "external_variables.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\n✓ Data saved to: {output_path}")
        print("\nFirst 5 rows:")
        print(final_df.head())
    else:
        print("\n✗ One or more datasets failed. Check errors above.")

    print("\nNOTE: Fuel prices (petrol, diesel, LPG) must be")
    print("added manually from NPA. Add as columns to the CSV.")
