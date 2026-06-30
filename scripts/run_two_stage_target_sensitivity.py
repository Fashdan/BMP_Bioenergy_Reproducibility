"""Two-stage BMP/VS -> BMP/DM sensitivity analysis.

This diagnostic uses already-saved repeated nested-CV out-of-fold predictions:

1. Take OOF predictions from the BMP per VS scenario.
2. Convert each prediction to BMP per DM using the measured VS/DM ratio.
3. Compare derived BMP/DM errors with the direct BMP/DM all-feature model.

No model is refit here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs"
PRED_DIR = OUT / "predictions"
TABLE_DIR = OUT / "tables"
VALIDATED_DIR = OUT / "validated_result_tables"

SOURCE_PREDICTIONS = PRED_DIR / "nested_cv_predictions_long.csv"

BMPVS_SCENARIO = "A_BMP_VS_all_features"
DIRECT_BMPDM_SCENARIO = "C_BMP_DM_all_features"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(obs, pred)))
    obs_mean = float(np.mean(obs))
    obs_range = float(np.max(obs) - np.min(obs))
    return {
        "n": int(len(obs)),
        "R2": float(r2_score(obs, pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(obs, pred)),
        "MAPE_pct": float(np.mean(np.abs((obs - pred) / obs)) * 100.0),
        "nRMSE_mean_pct": float(rmse / obs_mean * 100.0) if obs_mean else np.nan,
        "nRMSE_range_pct": float(rmse / obs_range * 100.0) if obs_range else np.nan,
        "RPD": float(np.std(obs, ddof=1) / rmse) if rmse else np.nan,
        "bias_pred_minus_obs": float(np.mean(pred - obs)),
        "median_abs_error": float(np.median(np.abs(obs - pred))),
    }


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "R2",
        "RMSE",
        "MAE",
        "MAPE_pct",
        "nRMSE_mean_pct",
        "nRMSE_range_pct",
        "RPD",
        "bias_pred_minus_obs",
        "median_abs_error",
    ]
    for keys, group in fold_metrics.groupby(["comparison_method", "model"], sort=False):
        comparison_method, model = keys
        row = {"comparison_method": comparison_method, "model": model, "n_folds": int(len(group))}
        for col in metric_cols:
            vals = group[col].dropna().to_numpy(dtype=float)
            row[f"{col}_fold_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{col}_fold_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            if len(vals) > 1:
                interval_low = np.quantile(vals, 0.025)
                interval_high = np.quantile(vals, 0.975)
            else:
                interval_low = interval_high = np.nan
            row[f"{col}_fold_interval_low"] = (
                float(interval_low) if not np.isnan(interval_low) else np.nan
            )
            row[f"{col}_fold_interval_high"] = (
                float(interval_high) if not np.isnan(interval_high) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    out = df[columns].copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out.to_markdown(index=False)


def main() -> None:
    ensure_dirs()
    predictions = pd.read_csv(SOURCE_PREDICTIONS)

    bmpvs = predictions[predictions["scenario_id"].eq(BMPVS_SCENARIO)].copy()
    direct = predictions[predictions["scenario_id"].eq(DIRECT_BMPDM_SCENARIO)].copy()

    bmpvs["comparison_method"] = "two_stage_BMPVS_OOF_times_measured_VS_DM"
    bmpvs["observed_BMP_DM"] = bmpvs["BMP_DM"]
    bmpvs["predicted_BMP_VS_oof"] = bmpvs["predicted"]
    bmpvs["predicted_BMP_DM_two_stage"] = bmpvs["predicted_BMP_VS_oof"] * bmpvs["VS_DM_ratio"]
    bmpvs["residual_obs_minus_pred_BMP_DM_two_stage"] = (
        bmpvs["observed_BMP_DM"] - bmpvs["predicted_BMP_DM_two_stage"]
    )
    bmpvs["error_pred_minus_obs_BMP_DM_two_stage"] = (
        bmpvs["predicted_BMP_DM_two_stage"] - bmpvs["observed_BMP_DM"]
    )
    bmpvs["abs_error_BMP_DM_two_stage"] = np.abs(bmpvs["residual_obs_minus_pred_BMP_DM_two_stage"])
    bmpvs["abs_pct_error_BMP_DM_two_stage"] = (
        bmpvs["abs_error_BMP_DM_two_stage"] / bmpvs["observed_BMP_DM"] * 100.0
    )

    derived_cols = [
        "comparison_method",
        "model",
        "outer_id",
        "repeat",
        "fold",
        "source_row_id",
        "row_index_filtered",
        "source_label",
        "Family",
        "Type",
        "Sub Type",
        "DM",
        "VS",
        "VS_DM_ratio",
        "BMP_VS",
        "observed_BMP_DM",
        "predicted_BMP_VS_oof",
        "predicted_BMP_DM_two_stage",
        "residual_obs_minus_pred_BMP_DM_two_stage",
        "error_pred_minus_obs_BMP_DM_two_stage",
        "abs_error_BMP_DM_two_stage",
        "abs_pct_error_BMP_DM_two_stage",
        "selected_params_json",
    ]
    derived = bmpvs[derived_cols].copy()

    merge_keys = ["model", "outer_id", "repeat", "fold", "source_row_id"]
    direct_keep = direct[
        merge_keys
        + [
            "predicted",
            "residual_obs_minus_pred",
            "error_pred_minus_obs",
            "abs_error",
            "abs_pct_error",
            "selected_params_json",
        ]
    ].rename(
        columns={
            "predicted": "predicted_BMP_DM_direct",
            "residual_obs_minus_pred": "residual_obs_minus_pred_BMP_DM_direct",
            "error_pred_minus_obs": "error_pred_minus_obs_BMP_DM_direct",
            "abs_error": "abs_error_BMP_DM_direct",
            "abs_pct_error": "abs_pct_error_BMP_DM_direct",
            "selected_params_json": "direct_selected_params_json",
        }
    )
    comparison = derived.merge(direct_keep, on=merge_keys, how="inner")
    comparison["abs_error_delta_two_stage_minus_direct"] = (
        comparison["abs_error_BMP_DM_two_stage"] - comparison["abs_error_BMP_DM_direct"]
    )
    comparison["squared_error_delta_two_stage_minus_direct"] = (
        comparison["residual_obs_minus_pred_BMP_DM_two_stage"] ** 2
        - comparison["residual_obs_minus_pred_BMP_DM_direct"] ** 2
    )

    fold_rows = []
    for keys, group in comparison.groupby(["model", "outer_id", "repeat", "fold"], sort=False):
        model, outer_id, repeat, fold = keys
        for method, pred_col in [
            ("two_stage_BMPVS_OOF_times_measured_VS_DM", "predicted_BMP_DM_two_stage"),
            ("direct_BMPDM_all_features", "predicted_BMP_DM_direct"),
        ]:
            row = {
                "comparison_method": method,
                "model": model,
                "outer_id": int(outer_id),
                "repeat": int(repeat),
                "fold": int(fold),
            }
            row.update(metrics(group["observed_BMP_DM"].to_numpy(), group[pred_col].to_numpy()))
            fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)
    metrics_summary = summarize_fold_metrics(fold_metrics)

    paired_summary_rows = []
    for model, group in comparison.groupby("model", sort=False):
        paired_summary_rows.append(
            {
                "model": model,
                "n_oof_predictions": int(len(group)),
                "mean_abs_error_two_stage": float(group["abs_error_BMP_DM_two_stage"].mean()),
                "mean_abs_error_direct": float(group["abs_error_BMP_DM_direct"].mean()),
                "mean_abs_error_delta_two_stage_minus_direct": float(
                    group["abs_error_delta_two_stage_minus_direct"].mean()
                ),
                "median_abs_error_delta_two_stage_minus_direct": float(
                    group["abs_error_delta_two_stage_minus_direct"].median()
                ),
                "two_stage_better_abs_error_rate": float(
                    (group["abs_error_BMP_DM_two_stage"] < group["abs_error_BMP_DM_direct"]).mean()
                ),
                "mean_squared_error_delta_two_stage_minus_direct": float(
                    group["squared_error_delta_two_stage_minus_direct"].mean()
                ),
            }
        )
    paired_summary = pd.DataFrame(paired_summary_rows)

    derived.to_csv(PRED_DIR / "two_stage_bmpvs_to_bmpdm_oof_predictions.csv", index=False)
    comparison.to_csv(PRED_DIR / "two_stage_vs_direct_bmpdm_oof_comparison.csv", index=False)
    fold_metrics.to_csv(TABLE_DIR / "two_stage_bmpvs_to_bmpdm_fold_metrics.csv", index=False)
    metrics_summary.to_csv(TABLE_DIR / "two_stage_bmpvs_to_bmpdm_metrics_summary.csv", index=False)
    paired_summary.to_csv(TABLE_DIR / "two_stage_vs_direct_bmpdm_paired_error_summary.csv", index=False)
    metrics_summary.to_csv(
        VALIDATED_DIR / "Table_validated_two_stage_bmpvs_to_bmpdm_sensitivity.csv", index=False
    )

    rf_summary = metrics_summary[metrics_summary["model"].eq("RF_nested_gridsearch")].copy()
    rf_paired = paired_summary[paired_summary["model"].eq("RF_nested_gridsearch")].copy()
    lines = [
        "# Two-Stage BMP/VS To BMP/DM Sensitivity",
        "",
        "## Purpose",
        "This diagnostic separates the machine-learning prediction of BMP per VS from the deterministic conversion to BMP per DM.",
        "No models were refit. The analysis reuses repeated nested-CV out-of-fold predictions from the BMP per VS scenario, multiplies each out-of-fold prediction by that row's measured VS/DM ratio, and compares the result with observed BMP per DM.",
        "",
        "## Random Forest Nested-CV Result",
        markdown_table(
            rf_summary,
            [
                "comparison_method",
                "model",
                "R2_fold_mean",
                "RMSE_fold_mean",
                "MAE_fold_mean",
                "MAPE_pct_fold_mean",
                "RPD_fold_mean",
                "bias_pred_minus_obs_fold_mean",
            ],
        ),
        "",
        "## Direct Versus Two-Stage Paired Error Summary",
        markdown_table(
            rf_paired,
            [
                "model",
                "n_oof_predictions",
                "mean_abs_error_two_stage",
                "mean_abs_error_direct",
                "mean_abs_error_delta_two_stage_minus_direct",
                "median_abs_error_delta_two_stage_minus_direct",
                "two_stage_better_abs_error_rate",
            ],
        ),
        "",
        "## All-Model Summary",
        markdown_table(
            metrics_summary,
            [
                "comparison_method",
                "model",
                "R2_fold_mean",
                "RMSE_fold_mean",
                "MAE_fold_mean",
                "bias_pred_minus_obs_fold_mean",
            ],
            max_rows=20,
        ),
        "",
        "## Interpretation",
        "- Similar two-stage derived BMP/DM error and direct BMP/DM error indicates that the VS/DM conversion accounts for part of the practical target behavior.",
        "- Materially weaker two-stage derived BMP/DM error indicates stronger support for the direct BMP/DM model in this internal validation setting.",
        "- The deterministic conversion is not independent biochemical prediction.",
        "",
        "## Saved Artifacts",
        "- Row-level two-stage OOF predictions: `analysis_outputs/predictions/two_stage_bmpvs_to_bmpdm_oof_predictions.csv`",
        "- Row-level direct comparison: `analysis_outputs/predictions/two_stage_vs_direct_bmpdm_oof_comparison.csv`",
        "- Fold metrics: `analysis_outputs/tables/two_stage_bmpvs_to_bmpdm_fold_metrics.csv`",
        "- Summary table: `analysis_outputs/tables/two_stage_bmpvs_to_bmpdm_metrics_summary.csv`",
        "- Paired error summary: `analysis_outputs/tables/two_stage_vs_direct_bmpdm_paired_error_summary.csv`",
    ]
    (ROOT / "TWO_STAGE_BMPVS_TO_BMPDM_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
