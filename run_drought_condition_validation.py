from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import set_style


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = (
    ROOT
    / "derived_data"
    / "experiment_results"
    / "spatial_block_buffer_cv"
    / "spatial_block_cv_predictions.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "derived_data" / "experiment_results" / "drought_condition_validation_spatial"
TIMESCALE_ORDER = ["SPEI-1", "SPEI-3", "SPEI-6", "SPEI-12", "SPEI-24"]
THRESHOLDS = [
    (-1.0, "moderate_or_worse"),
    (-1.5, "severe_or_worse"),
    (-2.0, "extreme"),
]
FIELD_CANDIDATES = {
    "timescale": ["timescale", "scale_label", "scale", "Timescale"],
    "fold": ["fold", "Fold"],
    "station_id": ["station_id", "station", "StationID"],
    "year": ["year", "Year"],
    "month": ["month", "Month"],
    "obs": ["obs", "observed", "y_true", "truth"],
    "pred": ["pred", "predicted", "y_pred", "prediction"],
}

# This validation evaluates the ability of the reconstructed SPEI dataset to
# identify monthly drought conditions under commonly used standardized
# drought-index thresholds. The thresholds are applied in a cumulative manner
# to assess moderate-or-worse, severe-or-worse, and extreme drought conditions.
# The default validation uses spatial-block out-of-fold predictions. Holding out
# station IDs alone should not be described as spatial independence unless a
# spatial block or distance buffer is also enforced.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate monthly drought-threshold detection from OOF predictions.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    lower_lookup = {col.lower(): col for col in df.columns}
    for key, candidates in FIELD_CANDIDATES.items():
        matched = None
        for candidate in candidates:
            if candidate in df.columns:
                matched = candidate
                break
            if candidate.lower() in lower_lookup:
                matched = lower_lookup[candidate.lower()]
                break
        if matched is None:
            raise KeyError(f"Required field '{key}' not found. Available columns: {list(df.columns)}")
        mapping[key] = matched
    return mapping


def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def compute_detection_metrics(obs: pd.Series, pred: pd.Series, threshold: float) -> dict[str, float | int]:
    y_true = obs.to_numpy(dtype=float) <= threshold
    y_pred = pred.to_numpy(dtype=float) <= threshold

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    far = safe_divide(fp, tp + fp)
    csi = safe_divide(tp, tp + fp + fn)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    return {
        "n_samples": int(len(obs)),
        "n_drought_obs": int(np.sum(y_true)),
        "n_drought_pred": int(np.sum(y_pred)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "CSI": csi,
        "Accuracy": accuracy,
    }


def round_metric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].round(4)
    return out


def save_plot_multi(fig: plt.Figure, out_stem: Path, dpi: int = 320) -> None:
    fig.savefig(out_stem.with_suffix(".jpg"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_threshold_minus1_barplot(main_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = main_df.copy()
    metrics = ["Precision", "Recall", "F1", "CSI"]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#9C755F"]
    x = np.arange(len(plot_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for idx, (metric, color) in enumerate(zip(metrics, colors, strict=True)):
        ax.bar(x + (idx - 1.5) * width, plot_df[metric], width=width, label=metric, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Timescale"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Drought-condition detection under SPEI <= -1.0")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    save_plot_multi(fig, output_dir / "drought_detection_threshold_minus1_barplot")


def make_multithreshold_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> None:
    heatmap_df = summary_df.pivot(index="timescale", columns="threshold", values="CSI").reindex(TIMESCALE_ORDER)
    threshold_labels = [f"{col:.1f}" for col in heatmap_df.columns]
    data = heatmap_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(data, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(threshold_labels)))
    ax.set_xticklabels(threshold_labels)
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(list(heatmap_df.index))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Timescale")
    ax.set_title("CSI across drought severity thresholds")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            label = "NaN" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#111827")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CSI")
    fig.tight_layout()
    save_plot_multi(fig, output_dir / "drought_detection_multithreshold_heatmap")


def main() -> None:
    args = parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input prediction file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    column_map = resolve_columns(df)
    work_df = df.rename(columns={v: k for k, v in column_map.items()}).copy()
    work_df = work_df[["timescale", "fold", "station_id", "year", "month", "obs", "pred"]]
    work_df["timescale"] = work_df["timescale"].astype(str)
    work_df["fold"] = pd.to_numeric(work_df["fold"], errors="coerce")
    work_df["obs"] = pd.to_numeric(work_df["obs"], errors="coerce")
    work_df["pred"] = pd.to_numeric(work_df["pred"], errors="coerce")
    work_df = work_df.dropna(subset=["timescale", "fold", "obs", "pred"]).copy()
    work_df["fold"] = work_df["fold"].astype(int)
    work_df = work_df[work_df["timescale"].isin(TIMESCALE_ORDER)].copy()

    by_fold_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for timescale in TIMESCALE_ORDER:
        ts_df = work_df[work_df["timescale"] == timescale].copy()
        if ts_df.empty:
            continue

        for threshold, drought_level in THRESHOLDS:
            full_metrics = compute_detection_metrics(ts_df["obs"], ts_df["pred"], threshold)
            summary_rows.append(
                {
                    "timescale": timescale,
                    "threshold": threshold,
                    "drought_level": drought_level,
                    **full_metrics,
                }
            )

            for fold in sorted(ts_df["fold"].unique()):
                fold_df = ts_df[ts_df["fold"] == fold]
                fold_metrics = compute_detection_metrics(fold_df["obs"], fold_df["pred"], threshold)
                by_fold_rows.append(
                    {
                        "timescale": timescale,
                        "fold": int(fold),
                        "threshold": threshold,
                        "drought_level": drought_level,
                        **fold_metrics,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    by_fold_df = pd.DataFrame(by_fold_rows)
    main_table_df = (
        summary_df[summary_df["threshold"] == -1.0][["timescale", "Precision", "Recall", "F1", "FAR", "CSI"]]
        .copy()
        .rename(columns={"timescale": "Timescale"})
    )
    main_table_df["Timescale"] = pd.Categorical(main_table_df["Timescale"], categories=TIMESCALE_ORDER, ordered=True)
    main_table_df = main_table_df.sort_values("Timescale").reset_index(drop=True)
    all_thresholds_main_df = summary_df[
        ["timescale", "threshold", "drought_level", "Precision", "Recall", "F1", "FAR", "CSI"]
    ].copy()
    all_thresholds_main_df["timescale"] = pd.Categorical(
        all_thresholds_main_df["timescale"], categories=TIMESCALE_ORDER, ordered=True
    )
    all_thresholds_main_df = all_thresholds_main_df.sort_values(
        ["timescale", "threshold"], ascending=[True, False]
    ).reset_index(drop=True)

    metric_cols = ["Precision", "Recall", "F1", "FAR", "CSI", "Accuracy"]
    summary_out = round_metric_columns(summary_df, metric_cols)
    by_fold_out = round_metric_columns(by_fold_df, metric_cols)
    main_table_out = round_metric_columns(main_table_df, ["Precision", "Recall", "F1", "FAR", "CSI"])
    all_thresholds_main_out = round_metric_columns(
        all_thresholds_main_df,
        ["Precision", "Recall", "F1", "FAR", "CSI"],
    )

    summary_path = args.output_dir / "drought_condition_validation_summary.csv"
    by_fold_path = args.output_dir / "drought_condition_validation_by_fold.csv"
    main_table_path = args.output_dir / "drought_condition_validation_main_table.csv"
    all_thresholds_path = args.output_dir / "drought_condition_validation_all_thresholds_main_table.csv"
    summary_out.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_fold_out.to_csv(by_fold_path, index=False, encoding="utf-8-sig")
    main_table_out.to_csv(main_table_path, index=False, encoding="utf-8-sig")
    all_thresholds_main_out.to_csv(all_thresholds_path, index=False, encoding="utf-8-sig")

    make_threshold_minus1_barplot(main_table_out, args.output_dir)
    make_multithreshold_heatmap(summary_out, args.output_dir)

    print("Table X. Drought-condition detection performance for moderate-or-worse drought conditions (SPEI <= -1.0).")
    print("")
    print("Timescale | Precision | Recall | F1-score | FAR | CSI")
    for _, row in main_table_out.iterrows():
        print(
            f"{row['Timescale']:<8} | "
            f"{row['Precision']:.4f} | "
            f"{row['Recall']:.4f} | "
            f"{row['F1']:.4f} | "
            f"{row['FAR']:.4f} | "
            f"{row['CSI']:.4f}"
        )

    print(f"\nSaved summary table to: {summary_path}")
    print(f"Saved fold table to: {by_fold_path}")
    print(f"Saved main table to: {main_table_path}")
    print(f"Saved all-threshold main table to: {all_thresholds_path}")


if __name__ == "__main__":
    main()
