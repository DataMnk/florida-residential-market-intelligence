"""Simple binary classification: will median sale price increase next month?

Florida county x property type x month Redfin data.
Does not modify the original CSV. Does not use future values as predictors.
Does not use random train/test splitting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("data/redfin_property_types_monthly_all_counties_2023_Jan_to_2026_Aug.csv")
OUTPUT_DIR = Path("outputs/classification")

PRICE = "MEDIAN SALE PRICE NSA ($)"
CLASS_TARGET = "PRICE_INCREASE_NEXT_MONTH"
TREE_MAX_DEPTH = 5
TRAIN_END = pd.Timestamp("2025-11-01")
PURGE_MONTH = pd.Timestamp("2025-12-01")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-07-01")

NUMERIC_FEATURES = [
    PRICE,
    "HOMES SOLD",
    "INVENTORY",
    "MEDIAN DAYS ON MARKET (DAYS)",
    "MONTHS OF SUPPLY",
    "AVERAGE SALE TO LIST RATIO (%)",
    "YEAR",
    "MONTH",
]
CATEGORICAL_FEATURES = ["PROPERTY TYPE", "METRO"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

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


def savefig(name: str) -> Path:
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")
    return path


def load_florida() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["STATE"] = df["REGION NAME"].map(parse_state)
    df["COUNTY"] = df["REGION NAME"].map(parse_county)
    fl = df[df["STATE"] == "FL"].copy()
    fl["PERIOD BEGIN"] = pd.to_datetime(fl["PERIOD BEGIN"], errors="coerce")
    fl["YEAR"] = fl["PERIOD BEGIN"].dt.year
    fl["MONTH"] = fl["PERIOD BEGIN"].dt.month
    fl["METRO"] = fl["METRO"].fillna("Non-Metro")
    print(f"Loaded: {DATA_PATH}")
    print("Original CSV was not modified.")
    print(f"Florida rows: {len(fl)}")
    return fl


def create_next_month_target(fl: pd.DataFrame) -> pd.DataFrame:
    """Label current month using the next calendar month in the same county x type series."""
    fl = fl.sort_values(["COUNTY", "PROPERTY TYPE", "PERIOD BEGIN"]).copy()
    grouped = fl.groupby(["COUNTY", "PROPERTY TYPE"], sort=False)
    fl["NEXT_PERIOD"] = grouped["PERIOD BEGIN"].shift(-1)
    fl["NEXT_PRICE"] = grouped[PRICE].shift(-1)

    current_month = fl["PERIOD BEGIN"].dt.to_period("M")
    next_month = fl["NEXT_PERIOD"].dt.to_period("M")
    is_next_calendar_month = next_month == (current_month + 1)

    n_has_following_obs = int(fl["NEXT_PERIOD"].notna().sum())
    n_gap = int((fl["NEXT_PERIOD"].notna() & ~is_next_calendar_month).sum())
    n_current_price_missing = int(fl[PRICE].isna().sum())
    n_next_price_missing_when_consecutive = int(
        (is_next_calendar_month & fl["NEXT_PRICE"].isna()).sum()
    )
    n_no_following_obs = int(fl["NEXT_PERIOD"].isna().sum())

    valid = (
        is_next_calendar_month
        & fl[PRICE].notna()
        & fl["NEXT_PRICE"].notna()
    )
    labeled = fl.loc[valid].copy()
    labeled[CLASS_TARGET] = (labeled["NEXT_PRICE"] > labeled[PRICE]).astype(int)

    print("Target construction:")
    print("  Sorted by COUNTY, PROPERTY TYPE, PERIOD BEGIN.")
    print("  Compared within COUNTY x PROPERTY TYPE only.")
    print("  Required following observation to be exactly the next calendar month.")
    print(f"  Florida rows: {len(fl)}")
    print(f"  Rows with a following observation in the same series: {n_has_following_obs}")
    print(f"  Following observations skipped because of a missing-month gap: {n_gap}")
    print(f"  Consecutive next-month rows with missing next price: {n_next_price_missing_when_consecutive}")
    print(f"  Rows with missing current price: {n_current_price_missing}")
    print(f"  Terminal rows with no following observation: {n_no_following_obs}")
    print(f"  Labeled rows before complete-case predictor filter: {len(labeled)}")
    return labeled


def complete_case(labeled: pd.DataFrame) -> pd.DataFrame:
    needed = FEATURES + [CLASS_TARGET, "PERIOD BEGIN", "COUNTY"]
    n_before = len(labeled)
    model_df = labeled.dropna(subset=needed).copy()
    print(f"Dropped incomplete predictor rows: {n_before - len(model_df)}")
    print(f"Modeling rows remaining: {len(model_df)}")
    print("Features used (current month only):")
    for col in FEATURES:
        print(f"  - {col}")
    print("Excluded from predictors: NEXT_PRICE, NEXT_PERIOD, future values,")
    print("ACTIVE LISTINGS, sale PPSF, new listing prices.")
    assert "NEXT_PRICE" not in FEATURES
    assert CLASS_TARGET not in FEATURES
    return model_df


def print_class_balance(model_df: pd.DataFrame, label: str) -> None:
    counts = model_df[CLASS_TARGET].value_counts().sort_index()
    print(f"\n{label} class counts:")
    for cls, n in counts.items():
        pct = 100.0 * n / len(model_df)
        name = "increase" if cls == 1 else "no increase"
        print(f"  class {cls} ({name}): {n} ({pct:.2f}%)")


def chronological_split(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier months, test on later months, with a one-month purge gap.

    November 2025 labels use December 2025 prices. December 2025 is excluded so
    no training label depends on a January 2026 (test) price.
    """
    months = pd.to_datetime(np.sort(model_df["PERIOD BEGIN"].unique()))
    print("Available labeled PERIOD BEGIN months:")
    print(f"  {months.min().date()} to {months.max().date()} ({len(months)} unique months)")

    train_df = model_df[model_df["PERIOD BEGIN"] <= TRAIN_END].copy()
    purge_df = model_df[model_df["PERIOD BEGIN"] == PURGE_MONTH].copy()
    test_df = model_df[
        (model_df["PERIOD BEGIN"] >= TEST_START) & (model_df["PERIOD BEGIN"] <= TEST_END)
    ].copy()

    print("Holdout rule: train through 2025-11; purge 2025-12; test 2026-01 through 2026-07.")
    print(f"Training date range: {train_df['PERIOD BEGIN'].min().date()} to {train_df['PERIOD BEGIN'].max().date()}")
    print(f"Purge month:         {PURGE_MONTH.date()} (excluded from train and test); n={len(purge_df)}")
    print(f"Testing date range:  {test_df['PERIOD BEGIN'].min().date()} to {test_df['PERIOD BEGIN'].max().date()}")
    print(f"Training unique months: {train_df['PERIOD BEGIN'].nunique()}")
    print(f"Testing unique months:  {test_df['PERIOD BEGIN'].nunique()}")
    print(f"Training row count: {len(train_df)}")
    print(f"Test row count: {len(test_df)}")

    assert pd.Timestamp(train_df["PERIOD BEGIN"].max()) == TRAIN_END
    assert pd.Timestamp(test_df["PERIOD BEGIN"].min()) == TEST_START
    assert pd.Timestamp(test_df["PERIOD BEGIN"].max()) == TEST_END
    assert not (train_df["PERIOD BEGIN"] == PURGE_MONTH).any()
    assert not (test_df["PERIOD BEGIN"] == PURGE_MONTH).any()
    # Last training labels use December 2025 prices, not test-period prices.
    assert train_df["NEXT_PERIOD"].max() < TEST_START
    print(
        "Leakage check: max training NEXT_PERIOD = "
        f"{pd.Timestamp(train_df['NEXT_PERIOD'].max()).date()} < test start {TEST_START.date()}."
    )
    return train_df, test_df


def logistic_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def tree_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                DecisionTreeClassifier(random_state=42, max_depth=TREE_MAX_DEPTH),
            ),
        ]
    )


def hgb_pipeline() -> Pipeline:
    """Same encoded feature space as the tree: one-hot categoricals, numeric passthrough."""
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]
    )


def metric_dict(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
    }


def time_aware_cv(pipeline_fn, train_df: pd.DataFrame, n_splits: int = 4) -> pd.DataFrame:
    """TimeSeriesSplit on unique months, with a one-month purge gap.

    gap=1 drops the month after the training block so a training label, which uses
    the following month's price, cannot depend on a validation-period price.
    """
    months = pd.to_datetime(np.sort(train_df["PERIOD BEGIN"].unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)
    rows = []
    print(
        f"TimeSeriesSplit on {len(months)} unique training months; "
        f"n_splits={n_splits}, gap=1 (one-month purge)."
    )
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(months), start=1):
        train_months = months[tr_idx]
        val_months = months[va_idx]
        gap_months = months[int(tr_idx.max()) + 1 : int(va_idx.min())]
        tr = train_df[train_df["PERIOD BEGIN"].isin(train_months)]
        va = train_df[train_df["PERIOD BEGIN"].isin(val_months)]
        assert tr["NEXT_PERIOD"].max() < va["PERIOD BEGIN"].min()
        pipe = pipeline_fn()
        pipe.fit(tr[FEATURES], tr[CLASS_TARGET])
        pred = pipe.predict(va[FEATURES])
        if va[CLASS_TARGET].nunique() < 2:
            auc = np.nan
        else:
            prob = pipe.predict_proba(va[FEATURES])[:, 1]
            auc = float(roc_auc_score(va[CLASS_TARGET], prob))
        acc = float(accuracy_score(va[CLASS_TARGET], pred))
        f1 = float(f1_score(va[CLASS_TARGET], pred, zero_division=0))
        gap_label = ", ".join(pd.Timestamp(m).strftime("%Y-%m") for m in gap_months) or "none"
        rows.append(
            {
                "fold": fold,
                "train_start": pd.Timestamp(train_months.min()).date(),
                "train_end": pd.Timestamp(train_months.max()).date(),
                "val_start": pd.Timestamp(val_months.min()).date(),
                "val_end": pd.Timestamp(val_months.max()).date(),
                "n_train": len(tr),
                "n_val": len(va),
                "Accuracy": acc,
                "F1": f1,
                "ROC-AUC": auc,
            }
        )
        print(
            f"  Fold {fold}: train {rows[-1]['train_start']} to {rows[-1]['train_end']} "
            f"(n={len(tr)}); purge {gap_label}; "
            f"val {rows[-1]['val_start']} to {rows[-1]['val_end']} "
            f"(n={len(va)}); acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}"
        )
    return pd.DataFrame(rows)


def plot_confusion(y_true, y_pred, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["0: no increase", "1: increase"],
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    savefig(filename)


def plot_roc(results: dict, filename: str) -> None:
    plt.figure(figsize=(7, 5.5))
    for name, payload in results.items():
        fpr, tpr, _ = roc_curve(payload["y_true"], payload["y_prob"])
        auc = payload["metrics"]["ROC-AUC"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on chronological test set")
    plt.legend(loc="lower right")
    savefig(filename)


def plot_comparison_table(table: pd.DataFrame, filename: str) -> None:
    display = table.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Accuracy Mean", "CV F1 Mean", "CV ROC-AUC Mean"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    fig, ax = plt.subplots(figsize=(13, 3.1))
    ax.axis("off")
    ax.set_title("Classification Comparison (purged chronological test set)")
    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(display.columns))))
    tbl.scale(1.15, 1.7)
    savefig(filename)


def fit_and_report(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = metric_dict(y_test, pred, prob)
    print("Held-out chronological test metrics (class 1 = price increase):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            pred,
            digits=4,
            zero_division=0,
            target_names=["0: no increase", "1: increase"],
        )
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    return {"name": name, "pred": pred, "prob": prob, "metrics": metrics}


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

    section("DATA AND TARGET")
    fl = load_florida()
    labeled = create_next_month_target(fl)
    model_df = complete_case(labeled)
    print(f"\nTotal modeling row count: {len(model_df)}")
    print_class_balance(model_df, "Full modeling sample")
    print(f"Unique counties: {model_df['COUNTY'].nunique()}")
    print("PROPERTY TYPE counts:")
    print(model_df["PROPERTY TYPE"].value_counts().to_string())

    section("CHRONOLOGICAL TRAIN / TEST SPLIT")
    train_df, test_df = chronological_split(model_df)
    print_class_balance(train_df, "Training")
    print_class_balance(test_df, "Test")

    X_train, y_train = train_df[FEATURES], train_df[CLASS_TARGET]
    X_test, y_test = test_df[FEATURES], test_df[CLASS_TARGET]

    section("MODEL A — LOGISTIC REGRESSION")
    print("Preprocessing: OneHotEncoder(handle_unknown='ignore') for PROPERTY TYPE and METRO;")
    print("StandardScaler for numeric features. All steps are inside the Pipeline.")
    log_result = fit_and_report(
        "Logistic Regression",
        logistic_pipeline(),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    section("MODEL B — DECISION TREE")
    print(f"DecisionTreeClassifier(random_state=42, max_depth={TREE_MAX_DEPTH}).")
    print("max_depth=5 is a simple pre-specified constraint to limit overfitting.")
    print("No hyperparameter search was performed. Numeric features are not scaled.")
    tree_result = fit_and_report(
        "Decision Tree",
        tree_pipeline(),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    section("MODEL C — HIST GRADIENT BOOSTING")
    print("HistGradientBoostingClassifier(random_state=42) with default sklearn settings.")
    print("Same encoded feature space as the tree: OneHotEncoder + numeric passthrough.")
    print("No hyperparameter search was performed.")
    hgb_result = fit_and_report(
        "HistGradientBoosting",
        hgb_pipeline(),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    section("TIME-AWARE CROSS-VALIDATION ON TRAINING MONTHS")
    print("Model A — Logistic Regression")
    cv_log = time_aware_cv(logistic_pipeline, train_df, n_splits=4)
    print("\nModel B — Decision Tree")
    cv_tree = time_aware_cv(tree_pipeline, train_df, n_splits=4)
    print("\nModel C — HistGradientBoosting")
    cv_hgb = time_aware_cv(hgb_pipeline, train_df, n_splits=4)
    cv_means = {
        "Logistic Regression": cv_log[["Accuracy", "F1", "ROC-AUC"]].mean(),
        "Decision Tree": cv_tree[["Accuracy", "F1", "ROC-AUC"]].mean(),
        "HistGradientBoosting": cv_hgb[["Accuracy", "F1", "ROC-AUC"]].mean(),
    }
    print("\nCV means (training months only, one-month purge gap):")
    for name, means in cv_means.items():
        print(
            f"  {name}: Accuracy {means['Accuracy']:.4f}, "
            f"F1 {means['F1']:.4f}, ROC-AUC {means['ROC-AUC']:.4f}"
        )

    section("MODEL COMPARISON")
    comparison_rows = []
    for result, cv_name in (
        (log_result, "Logistic Regression"),
        (tree_result, "Decision Tree"),
        (hgb_result, "HistGradientBoosting"),
    ):
        means = cv_means[cv_name]
        comparison_rows.append(
            {
                "Model": cv_name,
                "Accuracy": result["metrics"]["Accuracy"],
                "Precision": result["metrics"]["Precision"],
                "Recall": result["metrics"]["Recall"],
                "F1": result["metrics"]["F1"],
                "ROC-AUC": result["metrics"]["ROC-AUC"],
                "CV Accuracy Mean": means["Accuracy"],
                "CV F1 Mean": means["F1"],
                "CV ROC-AUC Mean": means["ROC-AUC"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    print(comparison.to_string(index=False))
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")

    section("PLOTS")
    plot_comparison_table(comparison, "01_model_comparison_table.png")
    plot_confusion(
        y_test,
        log_result["pred"],
        "Logistic Regression confusion matrix\nPurged chronological test set",
        "02_confusion_matrix_logistic_regression.png",
    )
    plot_confusion(
        y_test,
        tree_result["pred"],
        "Decision Tree confusion matrix\nPurged chronological test set",
        "03_confusion_matrix_decision_tree.png",
    )
    plot_roc(
        {
            "Logistic Regression": {
                "y_true": y_test,
                "y_prob": log_result["prob"],
                "metrics": log_result["metrics"],
            },
            "Decision Tree": {
                "y_true": y_test,
                "y_prob": tree_result["prob"],
                "metrics": tree_result["metrics"],
            },
            "HistGradientBoosting": {
                "y_true": y_test,
                "y_prob": hgb_result["prob"],
                "metrics": hgb_result["metrics"],
            },
        },
        "04_roc_curves.png",
    )
    plot_confusion(
        y_test,
        hgb_result["pred"],
        "HistGradientBoosting confusion matrix\nPurged chronological test set",
        "05_confusion_matrix_hist_gradient_boosting.png",
    )

    section("INTERPRETATION NOTES")
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for metric in metric_names:
        scores = {
            "Logistic Regression": log_result["metrics"][metric],
            "Decision Tree": tree_result["metrics"][metric],
            "HistGradientBoosting": hgb_result["metrics"][metric],
        }
        winner = max(scores, key=scores.get)
        detail = ", ".join(f"{k} {v:.4f}" for k, v in scores.items())
        print(f"  Higher {metric}: {winner} ({detail})")
    print("Precision, Recall, and F1 are reported for class 1 (price increase).")
    print("December 2025 was excluded as a purge gap so training labels do not use test-month prices.")
    print("This is a next-month direction forecast, not a causal housing-market claim.")
    print("No future prices or future market metrics were used as predictors.")


if __name__ == "__main__":
    main()
