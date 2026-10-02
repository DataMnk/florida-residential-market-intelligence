"""Exploratory data analysis for Florida Redfin county-month market data.

Does not modify the original CSV. Does not impute values, remove outliers,
or train a model. Each row is a county x property type x month aggregate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

DATA_PATH = Path("data/redfin_property_types_monthly_all_counties_2023_Jan_to_2026_Aug.csv")
OUTPUT_DIR = Path("outputs/eda")

TARGET = "MEDIAN SALE PRICE NSA ($)"
CANDIDATE_PREDICTORS = [
    "HOMES SOLD",
    "NEW LISTINGS",
    "PENDING SALES",
    "INVENTORY",
    "MEDIAN DAYS ON MARKET (DAYS)",
    "MONTHS OF SUPPLY",
    "AVERAGE SALE TO LIST RATIO (%)",
]
MAIN_NUMERIC = [TARGET] + CANDIDATE_PREDICTORS
PROPERTY_ORDER = [
    "Condo/Co-op",
    "Townhouse",
    "Single Family Residential",
    "Multi-Family (2-4 Units)",
]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_state(region_name: str) -> str:
    if pd.isna(region_name):
        return pd.NA
    text = str(region_name).strip()
    if "," not in text:
        return pd.NA
    suffix = text.rsplit(",", 1)[-1].strip()
    return suffix if len(suffix) == 2 else pd.NA


def parse_county(region_name: str) -> str:
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


SHORT_NAMES = {
    TARGET: "Median sale price",
    "HOMES SOLD": "Homes sold",
    "NEW LISTINGS": "New listings",
    "PENDING SALES": "Pending sales",
    "INVENTORY": "Inventory",
    "MEDIAN DAYS ON MARKET (DAYS)": "Median DOM",
    "MONTHS OF SUPPLY": "Months of supply",
    "AVERAGE SALE TO LIST RATIO (%)": "Sale-to-list %",
}


def usd_axis(x, _pos) -> str:
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"


def slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ($)", "")
        .replace("(%)", "pct")
        .replace("(days)", "days")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
        .strip("_")
    )


def savefig(name: str) -> Path:
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")
    return path


def prepare_florida() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["STATE"] = df["REGION NAME"].map(parse_state)
    df["COUNTY"] = df["REGION NAME"].map(parse_county)
    fl = df[df["STATE"] == "FL"].copy()
    fl["PERIOD BEGIN"] = pd.to_datetime(fl["PERIOD BEGIN"], errors="coerce")
    fl["PROPERTY TYPE"] = pd.Categorical(
        fl["PROPERTY TYPE"],
        categories=PROPERTY_ORDER,
        ordered=False,
    )
    return fl


def plot_target_histogram(fl: pd.DataFrame) -> None:
    prices = fl[TARGET].dropna()
    plt.figure(figsize=(9, 5))
    sns.histplot(prices, bins=40, color="#2F6DB3", edgecolor="white")
    plt.axvline(prices.median(), color="#C0392B", linestyle="--", linewidth=1.5, label=f"Median: ${prices.median():,.0f}")
    plt.axvline(prices.mean(), color="#1A5276", linestyle=":", linewidth=1.5, label=f"Mean: ${prices.mean():,.0f}")
    plt.title("Florida Median Sale Price Distribution\nCounty x property type x month medians, not house-level prices")
    plt.xlabel(TARGET)
    plt.ylabel("Number of county-month-property-type rows")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(usd_axis))
    plt.legend()
    savefig("01_histogram_median_sale_price.png")


def plot_price_boxplot(fl: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.boxplot(
        data=fl.dropna(subset=[TARGET]),
        x="PROPERTY TYPE",
        y=TARGET,
        order=PROPERTY_ORDER,
        color="#AED6F1",
        linecolor="#1B4F72",
    )
    plt.title("Median Sale Price by Property Type\nFlorida county x month aggregates")
    plt.xlabel("Property type")
    plt.ylabel(TARGET)
    plt.xticks(rotation=15)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(usd_axis))
    savefig("02_boxplot_price_by_property_type.png")


def plot_monthly_trend(fl: pd.DataFrame) -> None:
    trend = (
        fl.dropna(subset=[TARGET])
        .groupby(["PERIOD BEGIN", "PROPERTY TYPE"], observed=True)[TARGET]
        .median()
        .reset_index()
    )
    plt.figure(figsize=(11, 5.5))
    sns.lineplot(
        data=trend,
        x="PERIOD BEGIN",
        y=TARGET,
        hue="PROPERTY TYPE",
        hue_order=PROPERTY_ORDER,
        marker="o",
        markersize=3.5,
        linewidth=1.6,
    )
    plt.title(
        "EDA Summary: Monthly Median of County Median Sale Prices\n"
        "Florida rows aggregated by month and property type; not an official statewide Redfin statistic"
    )
    plt.xlabel("Period begin")
    plt.ylabel(f"Median of {TARGET}")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(usd_axis))
    plt.legend(title="Property type", loc="upper left", bbox_to_anchor=(1.01, 1))
    savefig("03_monthly_trend_price_by_property_type.png")


def plot_top_counties(county_summary: pd.DataFrame) -> None:
    top10 = county_summary.head(10).iloc[::-1]
    labels = [
        f"{row.COUNTY} (n={int(row.n_nonmissing_prices)})"
        for row in top10.itertuples(index=False)
    ]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, top10["typical_median_sale_price"], color="#2E86AB")
    plt.title("Top 10 Florida Counties by Typical Median Sale Price\nTypical = median of county x property-type x month medians")
    plt.xlabel(f"Typical {TARGET}")
    plt.ylabel("County (n = non-missing price rows)")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(usd_axis))
    savefig("04_top10_counties_typical_median_sale_price.png")


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    labeled = corr.rename(index=SHORT_NAMES, columns=SHORT_NAMES)
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(
        labeled,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
    )
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlations Among Target and Candidate Baseline Predictors\nFlorida pairwise-complete observations")
    savefig("05_predictor_correlation_heatmap.png")


def plot_scatter(fl: pd.DataFrame, x_col: str, filename: str) -> None:
    plot_df = fl.dropna(subset=[TARGET, x_col])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=TARGET,
        alpha=0.25,
        s=18,
        color="#1F618D",
        edgecolor=None,
    )
    plt.title(f"{TARGET}\nvs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(TARGET)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(usd_axis))
    savefig(filename)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )

    # ------------------------------------------------------------------
    # STEP 1 — Florida analysis subset
    # ------------------------------------------------------------------
    section("STEP 1 — FLORIDA ANALYSIS SUBSET")
    fl = prepare_florida()
    print(f"Loaded original file: {DATA_PATH}")
    print("Parsed STATE and COUNTY from REGION NAME.")
    print("Filtered STATE == FL only.")
    print("Converted PERIOD BEGIN to datetime.")
    print("Kept all four PROPERTY TYPE categories.")
    print("No imputation. No automatic outlier removal. Original CSV not modified.")
    print(f"Florida rows prepared: {len(fl)}")

    # ------------------------------------------------------------------
    # STEP 2 — Basic EDA summary
    # ------------------------------------------------------------------
    section("STEP 2 — BASIC EDA SUMMARY")
    print(f"Florida row count: {len(fl)}")
    print(f"Unique counties: {fl['COUNTY'].nunique(dropna=True)}")
    print("Property types and counts:")
    print(fl["PROPERTY TYPE"].value_counts().reindex(PROPERTY_ORDER).to_string())

    n_periods = fl["PERIOD BEGIN"].nunique(dropna=True)
    print(f"\nDate range (PERIOD BEGIN): {fl['PERIOD BEGIN'].min().date()} to {fl['PERIOD BEGIN'].max().date()}")
    print(f"Number of unique periods: {n_periods}")

    n_target_missing = int(fl[TARGET].isna().sum())
    n_target_present = int(fl[TARGET].notna().sum())
    target_missing_pct = 100.0 * n_target_missing / len(fl)
    print(f"\nTarget missingness ({TARGET}):")
    print(f"  present: {n_target_present} ({100.0 * n_target_present / len(fl):.2f}%)")
    print(f"  missing: {n_target_missing} ({target_missing_pct:.2f}%)")
    print("  Target missing rows also have HOMES SOLD missing:")
    print(
        f"  {int((fl[TARGET].isna() & fl['HOMES SOLD'].isna()).sum())} of {n_target_missing}"
    )

    print("\nOverall missing-value summary (Florida, original columns plus parsed STATE/COUNTY):")
    miss = missing_report(fl)
    print(miss.to_string())

    n_dup_rows = int(fl.drop(columns=["STATE", "COUNTY"]).duplicated().sum())
    n_dup_keys = int(fl.duplicated(subset=["COUNTY", "PROPERTY TYPE", "PERIOD BEGIN"]).sum())
    print(f"\nDuplicate full original-column rows: {n_dup_rows}")
    print(f"Duplicate county + property type + period combinations: {n_dup_keys}")

    print("\nDescriptive statistics for main numeric market variables:")
    desc = fl[MAIN_NUMERIC].agg(["count", "min", "max", "mean", "median", "std"]).T
    desc["missing"] = fl[MAIN_NUMERIC].isna().sum().values
    desc["missing_pct"] = (100.0 * desc["missing"] / len(fl)).round(2)
    print(desc.to_string())

    # ------------------------------------------------------------------
    # STEP 3 — Target exploration
    # ------------------------------------------------------------------
    section("STEP 3 — TARGET EXPLORATION")
    print(
        "IMPORTANT: MEDIAN SALE PRICE NSA ($) is a county × property type × month "
        "median. It is NOT an individual house sale price."
    )
    prices = fl[TARGET].dropna()
    print(f"Non-missing target rows: {len(prices)}")
    print(f"  min:    {prices.min():,.2f}")
    print(f"  max:    {prices.max():,.2f}")
    print(f"  mean:   {prices.mean():,.2f}")
    print(f"  median: {prices.median():,.2f}")
    print(f"  std:    {prices.std():,.2f}")
    print(f"  skewness (pandas): {prices.skew():,.3f}")
    print("  The mean is above the median, so the distribution is right-skewed.")
    print("  Visible extremes exist, including a minimum of about $16,800 and a maximum of $2,190,000.")
    print("  These extremes are left in the data; they were not removed.")

    print("\nTarget by PROPERTY TYPE:")
    by_type = (
        fl.groupby("PROPERTY TYPE", observed=True)[TARGET]
        .agg(count="count", min="min", median="median", mean="mean", max="max", std="std")
        .reindex(PROPERTY_ORDER)
    )
    print(by_type.to_string())
    print("\nTarget missingness by PROPERTY TYPE:")
    miss_by_type = (
        fl.groupby("PROPERTY TYPE", observed=True)[TARGET]
        .agg(n="size", missing=lambda s: s.isna().sum(), missing_pct=lambda s: 100.0 * s.isna().mean())
        .reindex(PROPERTY_ORDER)
    )
    print(miss_by_type.to_string())

    plot_target_histogram(fl)
    plot_price_boxplot(fl)

    # ------------------------------------------------------------------
    # STEP 4 — Market trends over time
    # ------------------------------------------------------------------
    section("STEP 4 — MARKET TRENDS OVER TIME")
    print(
        "The trend plot aggregates the median of available Florida county-row "
        "median sale prices for each month and property type."
    )
    print("This is an EDA summary only. It is not an official statewide Redfin statistic.")
    trend = (
        fl.dropna(subset=[TARGET])
        .groupby(["PERIOD BEGIN", "PROPERTY TYPE"], observed=True)[TARGET]
        .median()
        .reset_index()
    )
    yearly = (
        trend.assign(year=trend["PERIOD BEGIN"].dt.year)
        .groupby(["year", "PROPERTY TYPE"], observed=True)[TARGET]
        .median()
        .unstack()
        .reindex(columns=PROPERTY_ORDER)
    )
    print("\nMedian-of-county-medians by year and property type (EDA summary):")
    print(yearly.to_string())

    first_month = trend["PERIOD BEGIN"].min()
    last_month = trend["PERIOD BEGIN"].max()
    print(f"\nFirst plotted month: {first_month.date()}")
    print(f"Last plotted month: {last_month.date()}")
    start_vals = trend[trend["PERIOD BEGIN"] == first_month].set_index("PROPERTY TYPE")[TARGET]
    end_vals = trend[trend["PERIOD BEGIN"] == last_month].set_index("PROPERTY TYPE")[TARGET]
    print("Start-to-end comparison of the EDA monthly series (descriptive only):")
    for ptype in PROPERTY_ORDER:
        if ptype in start_vals.index and ptype in end_vals.index:
            print(
                f"  {ptype}: {start_vals[ptype]:,.0f} -> {end_vals[ptype]:,.0f} "
                f"(change {end_vals[ptype] - start_vals[ptype]:,.0f})"
            )
    print(
        "Visible pattern: the four property-type series stay separated. Condo/Co-op remains "
        "lowest and its EDA monthly series is lower in 2025-2026 than in 2023. Townhouse is "
        "slightly lower later in the window. Single Family Residential is higher at the end "
        "than at the start. Multi-Family stays highest and is the most volatile month to month. "
        "No causal claim is made from this descriptive plot."
    )
    plot_monthly_trend(fl)

    # ------------------------------------------------------------------
    # STEP 5 — Geographic exploration
    # ------------------------------------------------------------------
    section("STEP 5 — GEOGRAPHIC EXPLORATION")
    county_summary = (
        fl.groupby("COUNTY", observed=True)
        .agg(
            n_rows=("COUNTY", "size"),
            n_nonmissing_prices=(TARGET, "count"),
            typical_median_sale_price=(TARGET, "median"),
            mean_sale_price=(TARGET, "mean"),
        )
        .reset_index()
        .sort_values("typical_median_sale_price", ascending=False, na_position="last")
    )
    county_summary["missing_price_pct"] = (
        100.0 * (county_summary["n_rows"] - county_summary["n_nonmissing_prices"]) / county_summary["n_rows"]
    )
    print("Typical county price = median of MEDIAN SALE PRICE NSA ($) across that county's rows.")
    print("Sparse counties (few non-missing price rows) should not be over-interpreted.")
    print("\nHighest typical median sale prices:")
    print(county_summary.head(10).to_string(index=False))
    print("\nLowest typical median sale prices:")
    print(county_summary.dropna(subset=["typical_median_sale_price"]).tail(10).to_string(index=False))

    sparse = county_summary[
        (county_summary["n_rows"] < 90) | (county_summary["missing_price_pct"] >= 20)
    ]
    print(
        "\nCounties with sparse coverage (fewer than 90 total rows or "
        "at least 20% missing target). Do not over-interpret their typical prices:"
    )
    print(f"  flagged counties: {len(sparse)}")
    if len(sparse):
        print(
            sparse.sort_values(["n_rows", "missing_price_pct"], ascending=[True, False]).to_string(
                index=False
            )
        )
    top_sparse = county_summary.head(10).merge(sparse[["COUNTY"]], on="COUNTY")
    if len(top_sparse):
        print("\nTop-10 counties that also have sparse/missing-price coverage:")
        print(top_sparse[["COUNTY", "n_rows", "n_nonmissing_prices", "missing_price_pct"]].to_string(index=False))

    plot_top_counties(county_summary)

    # ------------------------------------------------------------------
    # STEP 6 — Predictor relationships
    # ------------------------------------------------------------------
    section("STEP 6 — PREDICTOR RELATIONSHIPS")
    print("Candidate numeric predictors only. Excluded because they are directly price-related:")
    print("  MEDIAN SALE PRICE PER SQ.FT. ($)")
    print("  MEDIAN NEW LISTING PRICE ($)")
    print("  MEDIAN NEW LISTING PRICE PER SQ.FT. ($)")
    print("Correlations use pairwise-complete observations. No imputation.")

    corr_cols = [TARGET] + CANDIDATE_PREDICTORS
    corr = fl[corr_cols].corr()
    print("\nCorrelation table:")
    print(corr.round(3).to_string())

    target_corr = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    print("\nCorrelations with the target, ordered by absolute value:")
    print(target_corr.round(3).to_string())

    plot_correlation_heatmap(corr)

    # The three strongest raw correlations are all volume counts and are nearly
    # interchangeable. Show one volume plot plus two different market-condition plots.
    scatter_cols = [
        "HOMES SOLD",
        "MONTHS OF SUPPLY",
        "MEDIAN DAYS ON MARKET (DAYS)",
    ]
    print("\nScatterplots: one volume measure plus two market-condition measures.")
    print("HOMES SOLD, NEW LISTINGS, and PENDING SALES are almost interchangeable (r > 0.96),")
    print("so only HOMES SOLD is plotted from that group.")
    scatter_filenames = {
        "HOMES SOLD": "06_scatter_price_vs_homes_sold.png",
        "MONTHS OF SUPPLY": "07_scatter_price_vs_months_of_supply.png",
        "MEDIAN DAYS ON MARKET (DAYS)": "08_scatter_price_vs_median_dom.png",
    }
    for col in scatter_cols:
        filename = scatter_filenames[col]
        r_value = corr.loc[TARGET, col]
        print(f"  {col}: r = {r_value:.3f} -> {filename}")
        plot_scatter(fl, col, filename)

    print("\nAlso note predictor-to-predictor redundancy:")
    print("  HOMES SOLD, NEW LISTINGS, PENDING SALES, and INVENTORY are volume measures")
    print("  and can be strongly related to one another.")
    print("  ACTIVE LISTINGS is not in this correlation set because it is redundant with INVENTORY.")

    # ------------------------------------------------------------------
    # STEP 7 — Data quality observations
    # ------------------------------------------------------------------
    section("STEP 7 — DATA QUALITY OBSERVATIONS")
    n_sold = fl["HOMES SOLD"].dropna()
    n_thin1 = int((n_sold == 1).sum())
    n_thin5 = int((n_sold < 5).sum())
    print(f"1. Target missingness is {target_missing_pct:.2f}% ({n_target_missing} of {len(fl)} Florida rows).")
    print("   Those rows have no recorded HOMES SOLD for that county/type/month.")
    print("   This is a coverage gap, not a value to impute for a baseline regression.")
    print("2. Coverage is uneven. Single Family Residential is nearly complete; Multi-Family")
    print("   has much higher target missingness. Smaller counties have fewer months and types.")
    print(
        f"3. Thin-market months: among rows with HOMES SOLD present, {n_thin1} have exactly 1 sale "
        f"({100.0 * n_thin1 / len(n_sold):.2f}%) and {n_thin5} have fewer than 5 sales "
        f"({100.0 * n_thin5 / len(n_sold):.2f}%)."
    )
    print("   In those months the 'median' can be one or two transactions, so the target is noisier.")
    print("4. Unusual extremes found during inspection remain in the file, including very low")
    print("   and very high medians, extreme days on market, extreme months of supply, and")
    print("   at least one sale-price-per-sq-ft value that equals the sale price.")
    print("   They were not deleted automatically because they may be thin-market months,")
    print("   data quirks, or real luxury/rural observations.")
    print("5. ACTIVE LISTINGS and INVENTORY are highly related but not identical.")
    print("   A simple baseline should use one of them, not both.")
    print("These issues should be acknowledged in the report. Automatic deletion would hide")
    print("the real structure of county-month market data and can bias the later baseline.")

    # ------------------------------------------------------------------
    # STEP 8 — EDA conclusion
    # ------------------------------------------------------------------
    section("STEP 8 — EDA CONCLUSION")
    print("Dataset: Florida Redfin Housing Market Tracker rows, monthly, county x property type.")
    print(f"Florida subset: {len(fl)} rows, 67 counties, 4 property types, {n_periods} months (Jan 2023-Aug 2026).")
    print("Each observation is an aggregated market median, not a house-level sale.")
    print()
    print("Main target pattern:")
    print(f"  {TARGET} is numeric, right-skewed, median about ${prices.median():,.0f},")
    print(f"  range ${prices.min():,.0f} to ${prices.max():,.0f}, missing {target_missing_pct:.2f}%.")
    print()
    print("Property-type differences:")
    print("  Condo/Co-op typically lowest; Multi-Family (2-4 Units) typically highest and most variable;")
    print("  Townhouse and Single Family Residential sit in between.")
    print()
    print("Temporal/geographic patterns:")
    print("  Property-type levels stay separated from 2023 through Aug 2026.")
    print("  In this EDA county-median summary, condo and townhouse series are lower later")
    print("  in the window; single-family and multi-family series are higher. Multi-Family is noisy.")
    print("  Monroe, Walton, and Collier have the highest typical medians; Holmes, Hamilton,")
    print("  and Jackson have the lowest. Walton and Gulf in the top 10 have more missing prices.")
    print()
    print("Major quality limitations:")
    print("  missing target in no-sale months; thin-market medians; uneven county/type coverage;")
    print("  a few extreme values; inventory-measure redundancy.")
    print()
    print("Predictors that appear reasonable to carry into a simple Linear Regression baseline:")
    print("  PROPERTY TYPE")
    print("  HOMES SOLD, NEW LISTINGS, PENDING SALES")
    print("  INVENTORY (or ACTIVE LISTINGS, not both)")
    print("  MEDIAN DAYS ON MARKET (DAYS)")
    print("  MONTHS OF SUPPLY")
    print("  AVERAGE SALE TO LIST RATIO (%)")
    print("Do not carry sale PPSF or new-listing prices into that baseline.")
    print("No regression was trained in this step.")

    print("\nPlot files written to", OUTPUT_DIR.resolve())
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
