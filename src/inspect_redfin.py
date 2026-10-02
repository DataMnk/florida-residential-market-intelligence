"""Minimal inspection/audit of Redfin Housing Market Tracker data.

Does not modify the original CSV. Does not drop rows/columns except for
temporary Florida filtering used only for inspection.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/redfin_property_types_monthly_all_counties_2023_Jan_to_2026_Aug.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_state(region_name: str) -> str:
    """Extract trailing 2-letter state abbreviation from 'County Name, ST'."""
    if pd.isna(region_name):
        return pd.NA
    text = str(region_name).strip()
    if "," not in text:
        return pd.NA
    suffix = text.rsplit(",", 1)[-1].strip()
    return suffix if len(suffix) == 2 else pd.NA


def parse_county(region_name: str) -> str:
    """Extract county/region text before the trailing ', ST'."""
    if pd.isna(region_name):
        return pd.NA
    text = str(region_name).strip()
    if "," not in text:
        return text
    return text.rsplit(",", 1)[0].strip()


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    missing_count = frame.isna().sum()
    missing_pct = 100.0 * missing_count / len(frame)
    return pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": missing_pct.round(2),
        }
    ).sort_values(["missing_pct", "missing_count"], ascending=False)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    section("1. LOAD CSV")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {DATA_PATH}")

    # ------------------------------------------------------------------
    # 2. Basic structure
    # ------------------------------------------------------------------
    section("2. BASIC STRUCTURE")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nExact column names:")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:2d}. {col}")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Identify role of each column
    # ------------------------------------------------------------------
    section("3. COLUMN ROLE IDENTIFICATION")
    print(
        "There is NO dedicated STATE column. State is embedded in REGION NAME "
        "(format observed: 'County Name, ST')."
    )
    print(
        "There is NO dedicated COUNTY column. County is embedded in REGION NAME."
    )
    print(
        "There is NO column literally named 'Median Sale Price'. "
        "The closest (and only) median sale-price column is 'MEDIAN SALE PRICE NSA ($)'."
    )
    print()
    role_map = {
        "state": "Not a separate column. Parsed from REGION NAME suffix (e.g. 'Miami-Dade County, FL' -> FL).",
        "county / region": "REGION NAME (full 'County, ST' string); REGION ID (numeric id); REGION TYPE (observed values below).",
        "property type": "PROPERTY TYPE",
        "period/date": "PERIOD BEGIN, PERIOD END; FREQUENCY; LAST UPDATED",
        "median sale price": "MEDIAN SALE PRICE NSA ($)",
        "price per square foot": "MEDIAN SALE PRICE PER SQ.FT. ($); also MEDIAN NEW LISTING PRICE PER SQ.FT. ($)",
        "homes sold": "HOMES SOLD",
        "new listings": "NEW LISTINGS",
        "active listings / inventory": "ACTIVE LISTINGS; INVENTORY",
        "pending sales": "PENDING SALES",
        "median days on market": "MEDIAN DAYS ON MARKET (DAYS)",
        "months of supply": "MONTHS OF SUPPLY",
        "sale-to-list ratio": "AVERAGE SALE TO LIST RATIO (%)",
        "other useful housing-market metrics": (
            "SHARE SOLD ABOVE ORIGINAL LIST (%); "
            "MEDIAN NEW LISTING PRICE ($); "
            "PERCENT OFF MARKET IN TWO WEEKS (%); "
            "METRO; IS SEASONALLY ADJUSTED"
        ),
    }
    for role, cols in role_map.items():
        print(f"- {role}: {cols}")

    print("\nUnique values for identifier / metadata columns:")
    for col in [
        "FREQUENCY",
        "REGION TYPE",
        "IS SEASONALLY ADJUSTED",
        "PROPERTY TYPE",
    ]:
        values = df[col].dropna().unique().tolist()
        print(f"  {col}: {values}")

    print(f"\nUnique LAST UPDATED values: {sorted(df['LAST UPDATED'].dropna().unique().tolist())}")
    print(f"Unique METRO count (full file): {df['METRO'].nunique(dropna=True)}")

    # ------------------------------------------------------------------
    # 4. Florida subset
    # ------------------------------------------------------------------
    section("4. FLORIDA SUBSET")
    df = df.copy()
    df["_state"] = df["REGION NAME"].map(parse_state)
    df["_county"] = df["REGION NAME"].map(parse_county)

    print("Observed unique state suffixes parsed from REGION NAME:")
    state_counts = df["_state"].value_counts(dropna=False)
    print(state_counts.to_string())

    fl = df[df["_state"] == "FL"].copy()
    print(f"\nNumber of Florida rows: {len(fl)}")
    print(f"Number of unique counties (from REGION NAME): {fl['_county'].nunique(dropna=True)}")
    print("Unique Florida counties:")
    for county in sorted(fl["_county"].dropna().unique().tolist()):
        print(f"  - {county}")

    print("\nList of property types (Florida):")
    fl_property_types = sorted(fl["PROPERTY TYPE"].dropna().unique().tolist())
    for ptype in fl_property_types:
        n = int((fl["PROPERTY TYPE"] == ptype).sum())
        print(f"  - {ptype}  (n={n})")

    fl["PERIOD BEGIN"] = pd.to_datetime(fl["PERIOD BEGIN"], errors="coerce")
    fl["PERIOD END"] = pd.to_datetime(fl["PERIOD END"], errors="coerce")
    n_periods = fl["PERIOD BEGIN"].nunique(dropna=True)
    print(f"\nNumber of unique time periods (PERIOD BEGIN): {n_periods}")
    print(f"Earliest PERIOD BEGIN: {fl['PERIOD BEGIN'].min()}")
    print(f"Latest PERIOD BEGIN: {fl['PERIOD BEGIN'].max()}")
    print(f"Earliest PERIOD END: {fl['PERIOD END'].min()}")
    print(f"Latest PERIOD END: {fl['PERIOD END'].max()}")
    print(f"Unique FREQUENCY values (Florida): {fl['FREQUENCY'].dropna().unique().tolist()}")
    print(f"Unique REGION TYPE values (Florida): {fl['REGION TYPE'].dropna().unique().tolist()}")
    print(
        "Unique IS SEASONALLY ADJUSTED values (Florida): "
        f"{fl['IS SEASONALLY ADJUSTED'].dropna().unique().tolist()}"
    )
    print(f"Number of unique Florida metros: {fl['METRO'].nunique(dropna=True)}")
    print("Florida metros:")
    for metro in sorted(fl["METRO"].dropna().unique().tolist()):
        print(f"  - {metro}")

    # Expected monthly coverage check (inspection only)
    expected_periods = pd.date_range("2023-01-01", "2026-08-01", freq="MS")
    observed_periods = pd.DatetimeIndex(sorted(fl["PERIOD BEGIN"].dropna().unique()))
    missing_periods = expected_periods.difference(observed_periods)
    extra_periods = observed_periods.difference(expected_periods)
    print(f"\nExpected monthly starts Jan 2023-Aug 2026: {len(expected_periods)}")
    print(f"Observed unique PERIOD BEGIN: {len(observed_periods)}")
    print(f"Missing monthly starts: {list(missing_periods) if len(missing_periods) else 'None'}")
    print(f"Extra monthly starts: {list(extra_periods) if len(extra_periods) else 'None'}")

    county_period_type_counts = (
        fl.groupby(["_county", "PROPERTY TYPE"], dropna=False)["PERIOD BEGIN"]
        .nunique()
        .reset_index(name="n_periods")
    )
    print("\nPeriod coverage by county x property type:")
    print(f"  Combinations with all {n_periods} periods: {(county_period_type_counts['n_periods'] == n_periods).sum()}")
    print(
        "  Combinations with fewer periods: "
        f"{(county_period_type_counts['n_periods'] < n_periods).sum()}"
    )
    print(f"  Min periods in a combination: {county_period_type_counts['n_periods'].min()}")
    print(f"  Max periods in a combination: {county_period_type_counts['n_periods'].max()}")

    # ------------------------------------------------------------------
    # 5. Data quality (Florida)
    # ------------------------------------------------------------------
    section("5. DATA QUALITY — FLORIDA SUBSET")
    print("Missing-value count and percentage for every original column:")
    orig_cols = [c for c in fl.columns if not c.startswith("_")]
    miss = missing_report(fl[orig_cols])
    print(miss.to_string())

    high_missing = miss[miss["missing_pct"] >= 20]
    print("\nColumns with very high missingness (>=20%):")
    if high_missing.empty:
        print("  None")
    else:
        print(high_missing.to_string())

    n_dup_rows = int(fl[orig_cols].duplicated().sum())
    print(f"\nDuplicate full rows: {n_dup_rows}")

    key_cols = ["REGION NAME", "PROPERTY TYPE", "PERIOD BEGIN"]
    n_dup_keys = int(fl.duplicated(subset=key_cols).sum())
    print(f"Duplicate combinations of county + property type + period: {n_dup_keys}")
    if n_dup_keys:
        print(fl.loc[fl.duplicated(subset=key_cols, keep=False), key_cols].head(20).to_string(index=False))

    numeric_cols = fl[orig_cols].select_dtypes(include="number").columns.tolist()
    print("\nNumeric columns:", numeric_cols)

    print("\nSuspicious zero values (count and % of Florida rows):")
    zero_rows = []
    for col in numeric_cols:
        n_zero = int((fl[col] == 0).sum())
        pct_zero = 100.0 * n_zero / len(fl)
        zero_rows.append((col, n_zero, pct_zero))
        print(f"  {col}: {n_zero} ({pct_zero:.2f}%)")

    print("\nNegative values (should be rare/impossible for market counts/prices):")
    for col in numeric_cols:
        n_neg = int((fl[col] < 0).sum())
        if n_neg:
            print(f"  {col}: {n_neg} negative values; min={fl[col].min()}")
        else:
            print(f"  {col}: none")

    print("\nObvious impossible / extreme-value checks:")
    price_col = "MEDIAN SALE PRICE NSA ($)"
    ppsf_col = "MEDIAN SALE PRICE PER SQ.FT. ($)"
    list_ppsf_col = "MEDIAN NEW LISTING PRICE PER SQ.FT. ($)"
    list_price_col = "MEDIAN NEW LISTING PRICE ($)"
    stl_col = "AVERAGE SALE TO LIST RATIO (%)"
    above_col = "SHARE SOLD ABOVE ORIGINAL LIST (%)"
    off_mkt_col = "PERCENT OFF MARKET IN TWO WEEKS (%)"
    dom_col = "MEDIAN DAYS ON MARKET (DAYS)"
    mos_col = "MONTHS OF SUPPLY"

    checks = [
        ("median sale price <= 0", fl[price_col] <= 0),
        ("median sale price > 5,000,000", fl[price_col] > 5_000_000),
        ("sale PPSF <= 0", fl[ppsf_col] <= 0),
        ("sale PPSF > 2,000", fl[ppsf_col] > 2_000),
        ("new listing price <= 0", fl[list_price_col] <= 0),
        ("new listing PPSF <= 0", fl[list_ppsf_col] <= 0),
        ("sale-to-list ratio < 50", fl[stl_col] < 50),
        ("sale-to-list ratio > 150", fl[stl_col] > 150),
        ("share sold above list < 0", fl[above_col] < 0),
        ("share sold above list > 100", fl[above_col] > 100),
        ("percent off market in 2 weeks < 0", fl[off_mkt_col] < 0),
        ("percent off market in 2 weeks > 100", fl[off_mkt_col] > 100),
        ("median DOM < 0", fl[dom_col] < 0),
        ("median DOM > 365", fl[dom_col] > 365),
        ("months of supply < 0", fl[mos_col] < 0),
        ("months of supply > 24", fl[mos_col] > 24),
        ("homes sold < 0", fl["HOMES SOLD"] < 0),
        ("new listings < 0", fl["NEW LISTINGS"] < 0),
        ("active listings < 0", fl["ACTIVE LISTINGS"] < 0),
        ("inventory < 0", fl["INVENTORY"] < 0),
        ("pending sales < 0", fl["PENDING SALES"] < 0),
        ("homes sold == 0 but median sale price present", (fl["HOMES SOLD"] == 0) & fl[price_col].notna()),
        ("inventory != active listings (non-null pairs)", fl["INVENTORY"].notna() & fl["ACTIVE LISTINGS"].notna() & (fl["INVENTORY"] != fl["ACTIVE LISTINGS"])),
    ]
    for label, mask in checks:
        n = int(mask.sum())
        print(f"  {label}: {n}")

    print("\nACTIVE LISTINGS vs INVENTORY comparison (Florida, non-null pairs):")
    both = fl.dropna(subset=["ACTIVE LISTINGS", "INVENTORY"])
    if len(both):
        equal = int((both["ACTIVE LISTINGS"] == both["INVENTORY"]).sum())
        print(f"  rows with both non-null: {len(both)}")
        print(f"  exactly equal: {equal}")
        print(f"  not equal: {len(both) - equal}")
        corr = both["ACTIVE LISTINGS"].corr(both["INVENTORY"])
        print(f"  correlation: {corr:.4f}")
        print("  describe ACTIVE LISTINGS:")
        print(both["ACTIVE LISTINGS"].describe().to_string())
        print("  describe INVENTORY:")
        print(both["INVENTORY"].describe().to_string())
    else:
        print("  No overlapping non-null rows.")

    print("\nPERIOD BEGIN vs PERIOD END span in days:")
    span = (fl["PERIOD END"] - fl["PERIOD BEGIN"]).dt.days
    print(span.describe().to_string())
    print(f"  unique span day counts: {sorted(span.dropna().unique().tolist())}")

    print("\nColumns that appear redundant or identifier-only:")
    print("  - LAST UPDATED: metadata about scrape/update, not a market feature.")
    print("  - FREQUENCY: constant if all rows are monthly.")
    print("  - REGION TYPE: constant if all rows are County.")
    print("  - IS SEASONALLY ADJUSTED: constant flag; price column already labeled NSA.")
    print("  - REGION ID: identifier for REGION NAME.")
    print("  - PERIOD END: determined by PERIOD BEGIN + month for monthly data.")
    print("  - ACTIVE LISTINGS vs INVENTORY: compare empirically above.")
    print("  - MEDIAN SALE PRICE PER SQ.FT. ($): mathematically related to sale price and home size; leakage risk if used as a predictor of sale price.")
    print("  - MEDIAN NEW LISTING PRICE ($) and PPSF: related listing-side prices; possible leakage/redundancy with sale price.")

    # ------------------------------------------------------------------
    # 6. Numeric summaries for main market variables
    # ------------------------------------------------------------------
    section("6. NUMERIC SUMMARIES — FLORIDA MAIN MARKET VARIABLES")
    main_numeric = [
        "HOMES SOLD",
        "MEDIAN SALE PRICE NSA ($)",
        "MEDIAN DAYS ON MARKET (DAYS)",
        "AVERAGE SALE TO LIST RATIO (%)",
        "SHARE SOLD ABOVE ORIGINAL LIST (%)",
        "NEW LISTINGS",
        "ACTIVE LISTINGS",
        "INVENTORY",
        "PENDING SALES",
        "MEDIAN NEW LISTING PRICE ($)",
        "MEDIAN NEW LISTING PRICE PER SQ.FT. ($)",
        "MEDIAN SALE PRICE PER SQ.FT. ($)",
        "MONTHS OF SUPPLY",
        "PERCENT OFF MARKET IN TWO WEEKS (%)",
    ]
    summary = fl[main_numeric].agg(["min", "max", "mean", "median", "std", "count"]).T
    summary["missing"] = fl[main_numeric].isna().sum().values
    print(summary.to_string())

    print("\nMedian sale price by PROPERTY TYPE (Florida):")
    print(
        fl.groupby("PROPERTY TYPE")[price_col]
        .agg(["count", "min", "median", "mean", "max"])
        .sort_values("median")
        .to_string()
    )

    print("\nLowest 10 median sale prices (Florida):")
    cols_show = ["REGION NAME", "PROPERTY TYPE", "PERIOD BEGIN", price_col, "HOMES SOLD", ppsf_col]
    print(fl.nsmallest(10, price_col)[cols_show].to_string(index=False))

    print("\nHighest 10 median sale prices (Florida):")
    print(fl.nlargest(10, price_col)[cols_show].to_string(index=False))

    print("\nRows with HOMES SOLD == 0 (Florida):")
    zeros_sold = fl[fl["HOMES SOLD"] == 0]
    print(f"  count: {len(zeros_sold)}")
    if len(zeros_sold):
        print(zeros_sold[cols_show].head(15).to_string(index=False))

    print("\nRows with target missing (Florida):")
    print(f"  {price_col} missing: {int(fl[price_col].isna().sum())}")

    # ------------------------------------------------------------------
    # 7. Target and predictor assessment
    # ------------------------------------------------------------------
    section("7. TARGET AND PREDICTOR ASSESSMENT")
    n_target_missing = int(fl[price_col].isna().sum())
    n_target_present = int(fl[price_col].notna().sum())
    print(f"Proposed numeric target column: {price_col}")
    print(f"  non-missing: {n_target_present} ({100 * n_target_present / len(fl):.2f}%)")
    print(f"  missing: {n_target_missing} ({100 * n_target_missing / len(fl):.2f}%)")
    print(f"  min={fl[price_col].min()}, median={fl[price_col].median()}, max={fl[price_col].max()}")
    print(
        "  Assessment: this is a numeric continuous market-level median, so it can be used "
        "as a baseline regression target. It is NOT a house-level sale price; each row is a "
        "county x property-type x month aggregate."
    )

    print("\nReasonable candidate predictors (to consider, not yet selected):")
    print("  - PROPERTY TYPE (categorical market segment)")
    print("  - HOMES SOLD, NEW LISTINGS, PENDING SALES (activity / demand-supply volume)")
    print("  - INVENTORY or ACTIVE LISTINGS (stock; likely redundant with each other)")
    print("  - MEDIAN DAYS ON MARKET (DAYS) (liquidity / time-to-sale)")
    print("  - MONTHS OF SUPPLY (inventory pressure)")
    print("  - AVERAGE SALE TO LIST RATIO (%) (market heat)")
    print("  - SHARE SOLD ABOVE ORIGINAL LIST (%) (market heat)")
    print("  - PERCENT OFF MARKET IN TWO WEEKS (%) (market speed)")
    print("  - simple time features derived from PERIOD BEGIN (year, month)")
    print("  - county or metro as geography (see section 9)")

    print("\nVariables that should NOT be used as predictors of median sale price:")
    print("  Target leakage / direct price dependence:")
    print("    - MEDIAN SALE PRICE PER SQ.FT. ($): another sale-price outcome; near-direct dependence.")
    print("    - MEDIAN NEW LISTING PRICE ($): listing-side price, highly related to sale price.")
    print("    - MEDIAN NEW LISTING PRICE PER SQ.FT. ($): listing-side price intensity.")
    print("  Identifier-only / constants:")
    print("    - LAST UPDATED, FREQUENCY, REGION TYPE, IS SEASONALLY ADJUSTED, REGION ID")
    print("    - PERIOD END if PERIOD BEGIN is already used")
    print("  Redundancy:")
    print("    - Keep only one of INVENTORY vs ACTIVE LISTINGS if they are nearly the same.")
    print("    - REGION NAME and METRO overlap geographically; using both can be redundant.")
    print("  Excessive missingness:")
    miss_sorted = miss.sort_values("missing_pct", ascending=False)
    print("    Columns ranked by missingness (Florida):")
    print(miss_sorted.to_string())

    # ------------------------------------------------------------------
    # 8. PROPERTY TYPE
    # ------------------------------------------------------------------
    section("8. PROPERTY TYPE AS CATEGORICAL PREDICTOR")
    print("PROPERTY TYPE should be kept as a categorical predictor for a baseline model.")
    print("It defines distinct housing-market segments (different typical prices and liquidity).")
    print("Florida categories:")
    vc = fl["PROPERTY TYPE"].value_counts(dropna=False)
    print(vc.to_string())
    print(f"\nMissing PROPERTY TYPE in Florida: {int(fl['PROPERTY TYPE'].isna().sum())}")

    # ------------------------------------------------------------------
    # 9. County encoding tradeoff
    # ------------------------------------------------------------------
    section("9. COUNTY ENCODING TRADEOFF (NO IMPLEMENTATION)")
    n_counties = fl["_county"].nunique(dropna=True)
    print(f"Unique Florida counties: {n_counties}")
    print("Options for a simple baseline model:")
    print("  Exclude: smallest model, but ignores large between-county price differences.")
    print(
        "  One-hot encode: captures county-level price levels, which is important in Florida, "
        f"but {n_counties} counties plus property types would create many dummy columns for an intro assignment."
    )
    print(
        "  Simpler alternative: use METRO (fewer categories) or omit geography in the very first "
        "baseline and add it only if needed. Do not use target encoding or clustering in Project 1."
    )
    print(f"Unique Florida metros: {fl['METRO'].nunique(dropna=True)}")
    print("County row counts:")
    print(fl["_county"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 10. Time/date in baseline model
    # ------------------------------------------------------------------
    section("10. TIME/DATE IN THE BASELINE MODEL (NO FEATURE ENGINEERING YET)")
    print("PERIOD BEGIN is the correct period identifier.")
    print("Using raw timestamps as a numeric predictor is not appropriate.")
    print("Simple useful derived features later (do not create them now):")
    print("  - year")
    print("  - month (seasonality)")
    print("A linear year term can capture a broad 2023-2026 trend; month can capture seasonality.")
    print("Do not use LAST UPDATED. Do not one-hot every year-month unless the model becomes too wide.")
    print("\nFlorida rows by year:")
    print(fl["PERIOD BEGIN"].dt.year.value_counts().sort_index().to_string())
    print("\nFlorida rows by month (pooled across years):")
    print(fl["PERIOD BEGIN"].dt.month.value_counts().sort_index().to_string())

    # ------------------------------------------------------------------
    # 11. Concise recommendation
    # ------------------------------------------------------------------
    section("11. CONCISE RECOMMENDATION")
    print("Proposed target:")
    print(f"  {price_col}")
    print("  Scope: Florida county x property-type x month rows only.")
    print()
    print("Proposed predictor list for an introductory baseline regression:")
    print("  Categorical: PROPERTY TYPE")
    print("  Numeric market: HOMES SOLD, NEW LISTINGS, PENDING SALES,")
    print("                  INVENTORY (or ACTIVE LISTINGS, not both),")
    print("                  MEDIAN DAYS ON MARKET (DAYS), MONTHS OF SUPPLY,")
    print("                  AVERAGE SALE TO LIST RATIO (%)")
    print("  Optional simple additions: year and month from PERIOD BEGIN;")
    print("                             METRO or a reduced geography encoding")
    print()
    print("Columns to exclude from predictors:")
    print("  LAST UPDATED, FREQUENCY, REGION TYPE, IS SEASONALLY ADJUSTED, REGION ID, PERIOD END")
    print("  MEDIAN SALE PRICE PER SQ.FT. ($)")
    print("  MEDIAN NEW LISTING PRICE ($)")
    print("  MEDIAN NEW LISTING PRICE PER SQ.FT. ($)")
    print("  SHARE SOLD ABOVE ORIGINAL LIST (%) and PERCENT OFF MARKET IN TWO WEEKS (%)")
    print("    if they are mostly redundant with sale-to-list / DOM or have higher missingness")
    print("  Do not use the target itself as a predictor.")
    print()
    print("Major data-quality issues:")
    print(f"  - No explicit STATE/COUNTY columns; must parse REGION NAME. Florida rows = {len(fl)}.")
    print(f"  - Duplicate county+type+period rows: {n_dup_keys}")
    print(f"  - Duplicate full rows: {n_dup_rows}")
    print(f"  - Target missing: {n_target_missing}")
    print("  - See missingness table and zero-value counts above.")
    print()
    print("Is cleaning necessary before EDA/modeling?")
    print("  Yes, light cleaning only: filter to Florida; parse county/state from REGION NAME;")
    print("  parse dates; decide how to handle missing target rows and high-missing columns.")
    print("  Do not drop columns from the original file. Do not impute yet unless EDA shows a simple need.")
    print()
    print("Recommended scope for Project 1:")
    print("  Florida-only, monthly, county-level, by property type.")
    print("  One baseline linear regression predicting median sale price from a small set of")
    print("  non-leaky market and categorical features. Keep EDA descriptive (trends, missingness,")
    print("  correlations, county/property-type differences). No advanced models.")


if __name__ == "__main__":
    main()
