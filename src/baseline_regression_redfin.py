"""Two simple Linear Regression baselines for Florida Redfin market data.

Does not modify the original CSV. Does not impute the target. Does not
remove outliers. Does not use regularized, nonlinear, or tuned models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/redfin_property_types_monthly_all_counties_2023_Jan_to_2026_Aug.csv")
OUTPUT_DIR = Path("outputs/regression")

TARGET = "MEDIAN SALE PRICE NSA ($)"
NUMERIC_FEATURES = [
    "HOMES SOLD",
    "INVENTORY",
    "MEDIAN DAYS ON MARKET (DAYS)",
    "MONTHS OF SUPPLY",
    "AVERAGE SALE TO LIST RATIO (%)",
    "YEAR",
    "MONTH",
]
CAT_A = ["PROPERTY TYPE"]
CAT_B = ["PROPERTY TYPE", "METRO"]
FEATURES_A = CAT_A + NUMERIC_FEATURES
FEATURES_B = CAT_B + NUMERIC_FEATURES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
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


def usd_axis(x, _pos) -> str:
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"


def savefig(name: str) -> Path:
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")
    return path


def prepare_florida_model_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["STATE"] = df["REGION NAME"].map(parse_state)
    df["COUNTY"] = df["REGION NAME"].map(parse_county)
    fl = df[df["STATE"] == "FL"].copy()
    fl["PERIOD BEGIN"] = pd.to_datetime(fl["PERIOD BEGIN"], errors="coerce")
    fl["YEAR"] = fl["PERIOD BEGIN"].dt.year
    fl["MONTH"] = fl["PERIOD BEGIN"].dt.month

    n_fl = len(fl)
    n_target_missing = int(fl[TARGET].isna().sum())
    fl = fl.dropna(subset=[TARGET]).copy()

    # METRO missing is recoded, not dropped.
    n_metro_missing = int(fl["METRO"].isna().sum())
    fl["METRO"] = fl["METRO"].fillna("Non-Metro")

    complete_cols = [TARGET, "PROPERTY TYPE"] + NUMERIC_FEATURES
    n_before_complete = len(fl)
    fl = fl.dropna(subset=complete_cols).copy()
    n_dropped_predictors = n_before_complete - len(fl)

    print(f"Loaded: {DATA_PATH}")
    print("Original CSV was not modified.")
    print(f"Florida rows: {n_fl}")
    print(f"Dropped missing target: {n_target_missing}")
    print(f"METRO missing after target drop, recoded to 'Non-Metro': {n_metro_missing}")
    print(f"Dropped incomplete predictor rows (complete-case, no imputation): {n_dropped_predictors}")
    print(f"Modeling rows remaining: {len(fl)}")
    print("PROPERTY TYPE categories kept:")
    print(fl["PROPERTY TYPE"].value_counts().to_string())
    print(f"Unique METRO categories including Non-Metro: {fl['METRO'].nunique()}")
    print(f"Non-Metro rows in modeling frame: {int((fl['METRO'] == 'Non-Metro').sum())}")
    return fl


def make_pipeline(categorical_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def encoded_feature_count(pipeline: Pipeline) -> int:
    return int(len(pipeline.named_steps["preprocess"].get_feature_names_out()))


def plot_pred_vs_actual(y_true, y_pred, title: str, filename: str, lims) -> None:
    plt.figure(figsize=(7.5, 6))
    plt.scatter(y_true, y_pred, alpha=0.25, s=16, color="#1F618D", linewidths=0)
    plt.plot(lims, lims, color="#C0392B", linestyle="--", linewidth=1.3, label="Perfect prediction")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(usd_axis))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(usd_axis))
    plt.title(title)
    plt.xlabel("Actual median sale price")
    plt.ylabel("Predicted median sale price")
    plt.legend(loc="upper left")
    savefig(filename)


def plot_residuals(y_true, y_pred, title: str, filename: str) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(y_pred, residuals, alpha=0.25, s=16, color="#1F618D", linewidths=0)
    plt.axhline(0, color="#C0392B", linestyle="--", linewidth=1.3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(usd_axis))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(usd_axis))
    plt.title(title)
    plt.xlabel("Predicted median sale price")
    plt.ylabel("Residual (actual - predicted)")
    savefig(filename)


def plot_comparison_table(rows: list[dict]) -> None:
    display = pd.DataFrame(
        [
            {
                "Model": row["Model"],
                "Train n": int(row["Train n"]),
                "Test n": int(row["Test n"]),
                "Features in": int(row["n_features_in"]),
                "Features out": int(row["n_features_encoded"]),
                "MAE": f"${row['MAE']:,.0f}",
                "RMSE": f"${row['RMSE']:,.0f}",
                "R2": f"{row['R2']:.3f}",
            }
            for row in rows
        ]
    )

    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.axis("off")
    ax.set_title("Linear Regression Baseline Comparison (same test set)")
    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(display.columns))))
    tbl.scale(1.2, 1.8)
    savefig("00_model_comparison_table.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    section("DATA PREPARATION")
    fl = prepare_florida_model_frame()
    print("Excluded from predictors: NEW LISTINGS, PENDING SALES,")
    print("MEDIAN SALE PRICE PER SQ.FT. ($), MEDIAN NEW LISTING PRICE ($),")
    print("MEDIAN NEW LISTING PRICE PER SQ.FT. ($).")
    print("YEAR and MONTH are numeric time features from PERIOD BEGIN.")

    X_a = fl[FEATURES_A]
    X_b = fl[FEATURES_B]
    y = fl[TARGET]

    X_a_train, X_a_test, y_train, y_test = train_test_split(
        X_a, y, test_size=0.20, random_state=42
    )
    X_b_train = X_b.loc[X_a_train.index]
    X_b_test = X_b.loc[X_a_test.index]

    print(f"\nTraining rows: {len(X_a_train)}")
    print(f"Test rows: {len(X_a_test)}")
    print("Same 80/20 split used for both models (random_state=42).")

    section("MODEL A — BASELINE WITHOUT GEOGRAPHY")
    print("Predictors:", FEATURES_A)
    print(f"Input features before encoding: {len(FEATURES_A)}")
    pipe_a = make_pipeline(CAT_A)
    pipe_a.fit(X_a_train, y_train)
    n_enc_a = encoded_feature_count(pipe_a)
    print(f"Encoded features after preprocessing: {n_enc_a}")
    print("Encoded feature names:")
    print("  " + "\n  ".join(pipe_a.named_steps["preprocess"].get_feature_names_out()))
    pred_a = pipe_a.predict(X_a_test)
    metrics_a = evaluate(y_test, pred_a)
    print(f"MAE:  ${metrics_a['MAE']:,.2f}")
    print(f"RMSE: ${metrics_a['RMSE']:,.2f}")
    print(f"R^2:  {metrics_a['R2']:.4f}")

    section("MODEL B — BASELINE WITH GEOGRAPHY (METRO)")
    print("Predictors:", FEATURES_B)
    print(f"Input features before encoding: {len(FEATURES_B)}")
    pipe_b = make_pipeline(CAT_B)
    pipe_b.fit(X_b_train, y_train)
    n_enc_b = encoded_feature_count(pipe_b)
    print(f"Encoded features after preprocessing: {n_enc_b}")
    print(f"METRO categories in training data: {X_b_train['METRO'].nunique()}")
    pred_b = pipe_b.predict(X_b_test)
    metrics_b = evaluate(y_test, pred_b)
    print(f"MAE:  ${metrics_b['MAE']:,.2f}")
    print(f"RMSE: ${metrics_b['RMSE']:,.2f}")
    print(f"R^2:  {metrics_b['R2']:.4f}")

    section("COMPARISON TABLE")
    comparison = pd.DataFrame(
        [
            {
                "Model": "A: no geography",
                "Train n": len(X_a_train),
                "Test n": len(X_a_test),
                "n_features_in": len(FEATURES_A),
                "n_features_encoded": n_enc_a,
                "MAE": metrics_a["MAE"],
                "RMSE": metrics_a["RMSE"],
                "R2": metrics_a["R2"],
            },
            {
                "Model": "B: with METRO",
                "Train n": len(X_b_train),
                "Test n": len(X_b_test),
                "n_features_in": len(FEATURES_B),
                "n_features_encoded": n_enc_b,
                "MAE": metrics_b["MAE"],
                "RMSE": metrics_b["RMSE"],
                "R2": metrics_b["R2"],
            },
        ]
    )
    print(comparison.to_string(index=False))
    print("\nChange from Model A to Model B (B minus A):")
    print(f"  MAE:  ${metrics_b['MAE'] - metrics_a['MAE']:,.2f}")
    print(f"  RMSE: ${metrics_b['RMSE'] - metrics_a['RMSE']:,.2f}")
    print(f"  R^2:  {metrics_b['R2'] - metrics_a['R2']:+.4f}")
    plot_comparison_table(comparison.to_dict(orient="records"))

    section("DIAGNOSTIC PLOTS")
    all_vals = np.concatenate([y_test.to_numpy(), pred_a, pred_b])
    pad = 0.03 * (all_vals.max() - all_vals.min())
    lims = (all_vals.min() - pad, all_vals.max() + pad)
    plot_pred_vs_actual(
        y_test,
        pred_a,
        "Model A: Predicted vs Actual\nLinear baseline without geography",
        "01_pred_vs_actual_model_a.png",
        lims,
    )
    plot_pred_vs_actual(
        y_test,
        pred_b,
        "Model B: Predicted vs Actual\nLinear baseline with METRO",
        "02_pred_vs_actual_model_b.png",
        lims,
    )
    plot_residuals(
        y_test,
        pred_a,
        "Model A: Residuals vs Predicted\nLinear baseline without geography",
        "03_residuals_vs_predicted_model_a.png",
    )
    plot_residuals(
        y_test,
        pred_b,
        "Model B: Residuals vs Predicted\nLinear baseline with METRO",
        "04_residuals_vs_predicted_model_b.png",
    )

    section("INTERPRETATION NOTES")
    better = "B" if metrics_b["MAE"] < metrics_a["MAE"] else "A"
    print(f"Lower test MAE: Model {better}.")
    print("Adding METRO lets the linear model capture metro-level price differences.")
    print("This is a descriptive baseline comparison, not a causal claim.")
    print("Weak linear correlations in EDA imply a simple linear model may remain limited.")
    print("Each row is a county x property type x month median, not a house price.")
    print("No regression beyond ordinary LinearRegression was used.")


if __name__ == "__main__":
    main()
