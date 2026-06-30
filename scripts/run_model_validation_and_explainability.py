"""Public reproducible analysis workflow for the BMP analysis.

This script reads the local analytical table, runs repeated nested CV with
the Random Forest hyperparameter grid, compares target and feature-set scenarios,
and computes residual/error summaries, prediction intervals, Lignin-DM
scope checks, and cross-model SHAP stability outputs.

Run from the repository root:

    python analysis_outputs/run_analysis_analysis.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional for the summary tables.
    plt = None
    sns = None

try:
    import shap
except Exception:  # pragma: no cover - script will continue and label SHAP unavailable.
    shap = None


SEED = 42
DM_FILTER = 15.0
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "table_complete.csv"
OUT = ROOT / "analysis_outputs"
TABLE_DIR = OUT / "tables"
PRED_DIR = OUT / "predictions"
SHAP_DIR = OUT / "shap"
FIG_DIR = OUT / "figures"
VALIDATED_DIR = OUT / "validated_result_tables"

RAW_DM = "DM Mean (% FM)"
RAW_VS = "VS Mean (% FM)"
RAW_RATIO = "VS/DM Mean (% FM)"
RAW_BMP_VS = "BMP exp Mean (Nm3 CH4/t VS)"

FEATURES = [
    RAW_DM,
    RAW_VS,
    "C/N",
    "Carbon Mean (% DM)",
    "Hydrogen Mean (% DM)",
    "Nitrogen Mean (% DM)",
    "Sulfur Mean (% DM)",
    "Oxygen Mean (% DM)",
    "Cellulose Mean (g/100g DM)",
    "Hemicelluloses Mean (g/100g DM)",
    "Lignin Mean (g/100g DM)",
]

SHORT = {
    RAW_DM: "DM",
    RAW_VS: "VS",
    "C/N": "C/N",
    "Carbon Mean (% DM)": "Carbon",
    "Hydrogen Mean (% DM)": "Hydrogen",
    "Nitrogen Mean (% DM)": "Nitrogen",
    "Sulfur Mean (% DM)": "Sulfur",
    "Oxygen Mean (% DM)": "Oxygen",
    "Cellulose Mean (g/100g DM)": "Cellulose",
    "Hemicelluloses Mean (g/100g DM)": "Hemicelluloses",
    "Lignin Mean (g/100g DM)": "Lignin",
}

RF_GRID = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [5, 7, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.7],
}

RF_SELECTED_PARAMS = {
    "n_estimators": 50,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "random_state": SEED,
    "n_jobs": 1,
}

GB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": SEED,
}

ET_PARAMS = {"n_estimators": 100, "random_state": SEED, "n_jobs": 1}


def parse_optional_depth(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "None", "nan", "NaN"}:
        return None
    return int(float(text))


def parse_max_features_value(value: Any) -> str | float | int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "None", "nan", "NaN"}:
            return None
        if text in {"sqrt", "log2"}:
            return text
        numeric = float(text)
        return int(numeric) if numeric.is_integer() else numeric
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    target_col: str
    target_label: str
    feature_cols: list[str]
    interpretation_note: str


@dataclass
class ManualSearchResult:
    best_estimator_: Pipeline
    best_params_: dict[str, Any]
    best_score_: float
    cv_results_: dict[str, Any]


def clean(name: str) -> str:
    return SHORT.get(name, name)


def ensure_dirs() -> None:
    for directory in [OUT, TABLE_DIR, PRED_DIR, SHAP_DIR, FIG_DIR, VALIDATED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_grid_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        out[key.replace("model__", "")] = value
    return out


def make_pipe(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def make_rf_pipe(params: dict[str, Any] | None = None) -> Pipeline:
    rf_params = dict(RF_SELECTED_PARAMS if params is None else params)
    rf_params.setdefault("random_state", SEED)
    rf_params.setdefault("n_jobs", 1)
    return make_pipe(RandomForestRegressor(**rf_params))


def prefix_forest_predict(model: RandomForestRegressor, x_model: np.ndarray, n_trees: int) -> np.ndarray:
    preds = np.asarray([tree.predict(x_model) for tree in model.estimators_[:n_trees]])
    return preds.mean(axis=0)


def evaluate_prefix_base_group(
    base_items: tuple[tuple[str, Any], ...],
    cand_list: list[tuple[int, dict[str, Any]]],
    x_train_model: np.ndarray,
    y_train_inner: np.ndarray,
    x_val_model: np.ndarray,
    y_val: np.ndarray,
) -> list[tuple[int, float, float]]:
    base_params = normalize_grid_params(dict(base_items))
    rf_params = dict(base_params)
    rf_params.update({"n_estimators": max(RF_GRID["model__n_estimators"]), "random_state": SEED, "n_jobs": 1})
    forest = RandomForestRegressor(**rf_params)
    forest.fit(x_train_model, y_train_inner)
    rows = []
    for cand_idx, cand in cand_list:
        n_trees = int(cand["model__n_estimators"])
        val_pred = prefix_forest_predict(forest, x_val_model, n_trees)
        train_pred = prefix_forest_predict(forest, x_train_model, n_trees)
        rows.append((cand_idx, r2_score(y_val, val_pred), r2_score(y_train_inner, train_pred)))
    return rows


def manual_complete_rf_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    inner: KFold,
    rf_jobs: int,
) -> ManualSearchResult:
    """Evaluate the Random Forest grid with a prefix-tree optimization.

    For each non-n_estimators hyperparameter combination, fitting a 150-tree
    forest produces the same first 50 and first 100 trees that sklearn would
    produce for the 50- and 100-tree selecteds with the same random_state.
    This preserves the selected grid while avoiding redundant tree fits.
    """

    selecteds = list(ParameterGrid(RF_GRID))
    base_groups: dict[tuple[tuple[str, Any], ...], list[tuple[int, dict[str, Any]]]] = {}
    for idx, cand in enumerate(selecteds):
        base_items = tuple((k, cand[k]) for k in sorted(cand) if k != "model__n_estimators")
        base_groups.setdefault(base_items, []).append((idx, cand))

    test_scores = [[] for _ in selecteds]
    train_scores = [[] for _ in selecteds]

    for train_idx, val_idx in inner.split(X, y):
        X_train_inner = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train_inner = y.iloc[train_idx].to_numpy(dtype=float)
        y_val = y.iloc[val_idx].to_numpy(dtype=float)

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train_imp = imputer.fit_transform(X_train_inner)
        x_train_model = scaler.fit_transform(x_train_imp)
        x_val_model = scaler.transform(imputer.transform(X_val))

        parallel_rows = Parallel(n_jobs=rf_jobs, prefer="processes")(
            delayed(evaluate_prefix_base_group)(
                base_items,
                cand_list,
                x_train_model,
                y_train_inner,
                x_val_model,
                y_val,
            )
            for base_items, cand_list in base_groups.items()
        )
        for rows in parallel_rows:
            for cand_idx, test_score, train_score in rows:
                test_scores[cand_idx].append(test_score)
                train_scores[cand_idx].append(train_score)

    mean_test = np.asarray([np.mean(v) for v in test_scores], dtype=float)
    std_test = np.asarray([np.std(v, ddof=0) for v in test_scores], dtype=float)
    mean_train = np.asarray([np.mean(v) for v in train_scores], dtype=float)
    std_train = np.asarray([np.std(v, ddof=0) for v in train_scores], dtype=float)
    ranks = stats.rankdata(-mean_test, method="min").astype(int)
    best_idx = int(np.flatnonzero(mean_test == np.max(mean_test))[0])
    best_params = normalize_grid_params(selecteds[best_idx])
    best_pipe = make_rf_pipe(
        {
            **best_params,
            "random_state": SEED,
            "n_jobs": 1,
        }
    )
    best_pipe.fit(X, y)
    cv_results = {
        "params": selecteds,
        "mean_test_score": mean_test,
        "std_test_score": std_test,
        "mean_train_score": mean_train,
        "std_train_score": std_train,
        "rank_test_score": ranks,
    }
    return ManualSearchResult(best_pipe, {f"model__{k}": v for k, v in best_params.items() if k not in {"random_state", "n_jobs"}}, float(mean_test[best_idx]), cv_results)


def fixed_models() -> dict[str, Pipeline]:
    return {
        "Mean_baseline": make_pipe(DummyRegressor(strategy="mean")),
        "LinearRegression": make_pipe(LinearRegression()),
        "Ridge_alpha_1": make_pipe(Ridge(alpha=1.0)),
        "RF_fixed_selected_params": make_rf_pipe(RF_SELECTED_PARAMS),
        "GradientBoosting_fixed": make_pipe(GradientBoostingRegressor(**GB_PARAMS)),
        "ExtraTrees_fixed": make_pipe(ExtraTreesRegressor(**ET_PARAMS)),
    }


def shap_models_for_scenario(rf_params: dict[str, Any] | None = None) -> dict[str, Pipeline]:
    return {
        "RandomForest": make_rf_pipe(rf_params or RF_SELECTED_PARAMS),
        "GradientBoosting": make_pipe(GradientBoostingRegressor(**GB_PARAMS)),
        "ExtraTrees": make_pipe(ExtraTreesRegressor(**ET_PARAMS)),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(DATA_PATH, na_values=["---"], thousands=",")
    df.insert(0, "source_row_id", np.arange(1, len(df) + 1))
    for col in FEATURES + [RAW_BMP_VS, RAW_RATIO, "Biodegradation %"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_label"] = (
        df["Family"].astype(str).fillna("")
        + " | "
        + df["Type"].astype(str).fillna("")
        + " | "
        + df["Sub Type"].astype(str).fillna("")
    )
    df["BMP_VS"] = df[RAW_BMP_VS]
    df["BMP_DM"] = df[RAW_BMP_VS] * df[RAW_RATIO]
    df["VS_DM_from_components"] = df[RAW_VS] / df[RAW_DM]
    df["BMP_DM_from_components"] = df[RAW_BMP_VS] * df["VS_DM_from_components"]
    df["BMP_DM_reported_ratio_delta"] = df["BMP_DM"] - (df[RAW_BMP_VS] * df[RAW_RATIO])
    df["VS_DM_rounding_delta"] = df[RAW_RATIO] - df["VS_DM_from_components"]
    df["included_DM_ge_15"] = df[RAW_DM] >= DM_FILTER
    required = FEATURES + ["BMP_DM", "BMP_VS"]
    df["complete_model_inputs"] = df[required].notna().all(axis=1)

    filtered = df[df["included_DM_ge_15"] & df["complete_model_inputs"]].copy()
    filtered = filtered.reset_index(drop=True)

    meta = {
        "data_path": str(DATA_PATH),
        "sha256": file_sha256(DATA_PATH),
        "raw_rows": int(len(df)),
        "filtered_rows": int(len(filtered)),
        "dm_filter": DM_FILTER,
        "excluded_dm_lt_15_rows": int((df[RAW_DM] < DM_FILTER).sum()),
        "raw_family_counts": df["Family"].value_counts(dropna=False).sort_index().to_dict(),
        "filtered_family_counts": filtered["Family"].value_counts(dropna=False).sort_index().to_dict(),
        "max_abs_bmp_dm_formula_delta": float(np.nanmax(np.abs(df["BMP_DM_reported_ratio_delta"]))),
        "max_abs_vs_dm_rounding_delta": float(np.nanmax(np.abs(df["VS_DM_rounding_delta"]))),
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    df.to_csv(TABLE_DIR / "raw_table_with_target_construction.csv", index=False)
    filtered.to_csv(TABLE_DIR / "analysis_table_filtered_dm_ge_15.csv", index=False)
    return df, filtered, meta


def build_scenarios() -> list[Scenario]:
    all_features = list(FEATURES)
    no_dm = [c for c in FEATURES if c != RAW_DM]
    no_vs = [c for c in FEATURES if c != RAW_VS]
    no_dm_vs = [c for c in FEATURES if c not in {RAW_DM, RAW_VS}]
    return [
        Scenario(
            "A_BMP_VS_all_features",
            "BMP_VS",
            "BMP per VS",
            all_features,
            "Original literature-normalized BMP target; avoids algebraic conversion to per-DM target.",
        ),
        Scenario(
            "B_BMP_DM_no_DM_VS",
            "BMP_DM",
            "BMP per DM",
            no_dm_vs,
            "Conservative BMP per DM scenario excluding the two variables that reconstruct VS/DM.",
        ),
        Scenario(
            "C_BMP_DM_all_features",
            "BMP_DM",
            "BMP per DM",
            all_features,
            "Primary BMP per DM setting with mathematically coupled DM and VS predictors.",
        ),
        Scenario(
            "D1_BMP_DM_no_DM",
            "BMP_DM",
            "BMP per DM",
            no_dm,
            "Single-variable exclusion sensitivity: DM removed, VS included.",
        ),
        Scenario(
            "D2_BMP_DM_no_VS",
            "BMP_DM",
            "BMP per DM",
            no_vs,
            "Single-variable exclusion sensitivity: VS removed, DM included.",
        ),
    ]


def scenario_by_id(scenarios: list[Scenario], scenario_ids: list[str] | None) -> list[Scenario]:
    if not scenario_ids:
        return scenarios
    keep = set(scenario_ids)
    unknown = keep.difference({s.scenario_id for s in scenarios})
    if unknown:
        raise ValueError(f"Unknown scenario(s): {sorted(unknown)}")
    return [s for s in scenarios if s.scenario_id in keep]


def build_outer_splits(n: int, n_repeats: int, n_splits: int, max_outer: int | None = None) -> list[dict[str, Any]]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    splits = []
    for idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n)), start=1):
        if max_outer is not None and idx > max_outer:
            break
        splits.append(
            {
                "outer_id": idx,
                "repeat": int((idx - 1) // n_splits + 1),
                "fold": int((idx - 1) % n_splits + 1),
                "train_idx": np.asarray(train_idx),
                "test_idx": np.asarray(test_idx),
            }
        )
    return splits


def save_fold_assignments(filtered: pd.DataFrame, splits: list[dict[str, Any]]) -> None:
    rows = []
    for split in splits:
        for idx in split["test_idx"]:
            rows.append(
                {
                    "source_row_id": int(filtered.loc[idx, "source_row_id"]),
                    "row_index_filtered": int(idx),
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "outer_id": split["outer_id"],
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "repeated_5x10_fold_assignments.csv", index=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def cv_results_to_rows(
    cv_results: dict[str, Any], scenario: Scenario, split: dict[str, Any], elapsed_s: float
) -> list[dict[str, Any]]:
    rows = []
    n = len(cv_results["params"])
    for i in range(n):
        params = normalize_grid_params(cv_results["params"][i])
        row = {
            "scenario_id": scenario.scenario_id,
            "outer_id": split["outer_id"],
            "repeat": split["repeat"],
            "fold": split["fold"],
            "selected_rank": int(cv_results["rank_test_score"][i]),
            "mean_test_r2": float(cv_results["mean_test_score"][i]),
            "std_test_r2": float(cv_results["std_test_score"][i]),
            "mean_train_r2": float(cv_results.get("mean_train_score", [np.nan] * n)[i]),
            "std_train_r2": float(cv_results.get("std_train_score", [np.nan] * n)[i]),
            "elapsed_grid_seconds_for_outer_fold": elapsed_s,
        }
        row.update(params)
        rows.append(row)
    return rows


def run_nested_cv(
    filtered: pd.DataFrame,
    scenarios: list[Scenario],
    splits: list[dict[str, Any]],
    grid_jobs: int,
    optimized_grid: bool = True,
    force: bool = False,
) -> None:
    pred_path = PRED_DIR / "nested_cv_predictions_long.csv"
    params_path = TABLE_DIR / "nested_cv_selected_hyperparameters.csv"
    selecteds_path = TABLE_DIR / "nested_gridsearch_selected_results_by_outer.csv"

    if force:
        for p in [pred_path, params_path, selecteds_path]:
            if p.exists():
                p.unlink()

    existing_pred = read_csv_if_exists(pred_path)
    done = set()
    if not existing_pred.empty:
        done = set(zip(existing_pred["scenario_id"], existing_pred["outer_id"], existing_pred["model"]))

    inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_selected_count = int(np.prod([len(v) for v in RF_GRID.values()]))

    for scenario in scenarios:
        X = filtered[scenario.feature_cols].copy()
        y = filtered[scenario.target_col].astype(float).copy()
        for split in splits:
            fold_key = (scenario.scenario_id, split["outer_id"])
            all_models_done = all((fold_key[0], fold_key[1], m) in done for m in ["RF_nested_gridsearch", *fixed_models().keys()])
            if all_models_done:
                continue

            X_train, X_test = X.iloc[split["train_idx"]], X.iloc[split["test_idx"]]
            y_train, y_test = y.iloc[split["train_idx"]], y.iloc[split["test_idx"]]

            fold_pred_rows: list[dict[str, Any]] = []
            fold_param_rows: list[dict[str, Any]] = []
            fold_selected_rows: list[dict[str, Any]] = []

            if (fold_key[0], fold_key[1], "RF_nested_gridsearch") not in done:
                t0 = time.time()
                if optimized_grid:
                    search = manual_complete_rf_grid_search(X_train, y_train, inner, rf_jobs=grid_jobs)
                    tuning_engine = "complete_grid_prefix_forest_equivalent"
                else:
                    estimator = make_pipe(RandomForestRegressor(random_state=SEED, n_jobs=1))
                    search = GridSearchCV(
                        estimator=estimator,
                        param_grid=RF_GRID,
                        scoring="r2",
                        cv=inner,
                        refit=True,
                        n_jobs=grid_jobs,
                        return_train_score=True,
                        error_score="raise",
                    )
                    search.fit(X_train, y_train)
                    tuning_engine = "sklearn_GridSearchCV"
                elapsed = time.time() - t0
                pred = search.best_estimator_.predict(X_test)
                best_params = normalize_grid_params(search.best_params_)

                fold_param_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "outer_id": split["outer_id"],
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "inner_cv": "KFold(n_splits=5, shuffle=True, random_state=42)",
                        "scoring": "r2",
                        "refit": True,
                        "tuning_engine": tuning_engine,
                        "selected_count": grid_selected_count,
                        "best_inner_r2": float(search.best_score_),
                        "elapsed_seconds": elapsed,
                        **best_params,
                    }
                )
                fold_selected_rows.extend(cv_results_to_rows(search.cv_results_, scenario, split, elapsed))
                fold_pred_rows.extend(
                    prediction_rows(filtered, scenario, split, y_test, pred, "RF_nested_gridsearch", best_params)
                )

            for model_name, pipe in fixed_models().items():
                if (fold_key[0], fold_key[1], model_name) in done:
                    continue
                fitted = clone(pipe)
                fitted.fit(X_train, y_train)
                pred = fitted.predict(X_test)
                fold_pred_rows.extend(prediction_rows(filtered, scenario, split, y_test, pred, model_name, {}))

            append_rows(pred_path, fold_pred_rows)
            append_rows(params_path, fold_param_rows)
            append_rows(selecteds_path, fold_selected_rows)
            print(
                f"completed {scenario.scenario_id} outer {split['outer_id']} "
                f"({split['repeat']}/{split['fold']})",
                flush=True,
            )


def prediction_rows(
    filtered: pd.DataFrame,
    scenario: Scenario,
    split: dict[str, Any],
    y_test: pd.Series,
    pred: np.ndarray,
    model_name: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for pos, row_idx in enumerate(split["test_idx"]):
        obs = float(y_test.iloc[pos])
        yhat = float(pred[pos])
        base = filtered.iloc[row_idx]
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "target_label": scenario.target_label,
                "model": model_name,
                "outer_id": split["outer_id"],
                "repeat": split["repeat"],
                "fold": split["fold"],
                "source_row_id": int(base["source_row_id"]),
                "row_index_filtered": int(row_idx),
                "source_label": base["source_label"],
                "Family": base["Family"],
                "Type": base["Type"],
                "Sub Type": base["Sub Type"],
                "observed": obs,
                "predicted": yhat,
                "residual_obs_minus_pred": obs - yhat,
                "error_pred_minus_obs": yhat - obs,
                "abs_error": abs(obs - yhat),
                "abs_pct_error": abs(obs - yhat) / obs * 100.0 if obs != 0 else np.nan,
                "DM": float(base[RAW_DM]),
                "VS": float(base[RAW_VS]),
                "VS_DM_ratio": float(base[RAW_RATIO]),
                "BMP_VS": float(base["BMP_VS"]),
                "BMP_DM": float(base["BMP_DM"]),
                "Lignin": float(base["Lignin Mean (g/100g DM)"]),
                "selected_params_json": json.dumps(params, sort_keys=True),
            }
        )
    return rows


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid_obs_minus_pred = y_true - y_pred
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom_mean = np.nanmean(y_true)
    denom_range = np.nanmax(y_true) - np.nanmin(y_true)
    return {
        "n": float(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE_pct": float(np.nanmean(np.abs(resid_obs_minus_pred / y_true)) * 100.0),
        "nRMSE_mean_pct": float(rmse / denom_mean * 100.0) if denom_mean else np.nan,
        "nRMSE_range_pct": float(rmse / denom_range * 100.0) if denom_range else np.nan,
        "RPD": float(np.nanstd(y_true, ddof=1) / rmse) if rmse else np.inf,
        "bias_pred_minus_obs": float(np.nanmean(y_pred - y_true)),
        "median_abs_error": float(np.nanmedian(np.abs(resid_obs_minus_pred))),
    }


def summarize_predictions(filtered: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred = pd.read_csv(PRED_DIR / "nested_cv_predictions_long.csv")
    fold_rows = []
    for keys, group in pred.groupby(["scenario_id", "model", "outer_id", "repeat", "fold"], dropna=False):
        scenario_id, model, outer_id, repeat, fold = keys
        metrics = metric_dict(group["observed"], group["predicted"])
        fold_rows.append(
            {
                "scenario_id": scenario_id,
                "model": model,
                "outer_id": outer_id,
                "repeat": repeat,
                "fold": fold,
                **metrics,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(TABLE_DIR / "nested_cv_fold_metrics.csv", index=False)

    summary_rows = []
    metric_cols = ["R2", "RMSE", "MAE", "MAPE_pct", "nRMSE_mean_pct", "nRMSE_range_pct", "RPD", "bias_pred_minus_obs"]
    for (scenario_id, model), group in fold_metrics.groupby(["scenario_id", "model"], dropna=False):
        pred_group = pred[(pred["scenario_id"] == scenario_id) & (pred["model"] == model)]
        pooled = metric_dict(pred_group["observed"], pred_group["predicted"])
        row_mean = (
            pred_group.groupby(["scenario_id", "model", "source_row_id"], as_index=False)
            .agg(observed=("observed", "first"), predicted=("predicted", "mean"))
        )
        row_metrics = metric_dict(row_mean["observed"], row_mean["predicted"])
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "model": model,
            "n_outer_folds": int(group["outer_id"].nunique()),
            "n_repeated_predictions": int(len(pred_group)),
            "n_unique_rows": int(pred_group["source_row_id"].nunique()),
        }
        for metric in metric_cols:
            vals = group[metric].dropna().to_numpy()
            row[f"{metric}_fold_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{metric}_fold_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{metric}_fold_ci_low"] = float(np.quantile(vals, 0.025)) if len(vals) else np.nan
            row[f"{metric}_fold_ci_high"] = float(np.quantile(vals, 0.975)) if len(vals) else np.nan
            row[f"{metric}_pooled_repeated"] = pooled[metric]
            row[f"{metric}_row_mean_prediction"] = row_metrics[metric]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["scenario_id", "R2_fold_mean"], ascending=[True, False])
    summary.to_csv(TABLE_DIR / "nested_cv_metrics_summary.csv", index=False)
    summary.to_csv(VALIDATED_DIR / "Table_validated_nested_cv_metrics_summary.csv", index=False)

    row_summary = (
        pred.groupby(["scenario_id", "model", "source_row_id"], as_index=False)
        .agg(
            target_label=("target_label", "first"),
            source_label=("source_label", "first"),
            Family=("Family", "first"),
            Type=("Type", "first"),
            **{"Sub Type": ("Sub Type", "first")},
            observed=("observed", "first"),
            mean_oof_prediction=("predicted", "mean"),
            sd_oof_prediction=("predicted", "std"),
            min_oof_prediction=("predicted", "min"),
            max_oof_prediction=("predicted", "max"),
            n_oof_predictions=("predicted", "count"),
            DM=("DM", "first"),
            VS=("VS", "first"),
            VS_DM_ratio=("VS_DM_ratio", "first"),
            BMP_VS=("BMP_VS", "first"),
            BMP_DM=("BMP_DM", "first"),
            Lignin=("Lignin", "first"),
        )
    )
    row_summary["residual_obs_minus_mean_oof"] = row_summary["observed"] - row_summary["mean_oof_prediction"]
    row_summary["error_mean_oof_minus_obs"] = row_summary["mean_oof_prediction"] - row_summary["observed"]
    row_summary["abs_error_mean_oof"] = row_summary["residual_obs_minus_mean_oof"].abs()
    row_summary["abs_pct_error_mean_oof"] = row_summary["abs_error_mean_oof"] / row_summary["observed"] * 100.0
    row_summary.to_csv(PRED_DIR / "nested_cv_row_level_oof_summary.csv", index=False)

    range_summary = summarize_ranges(pred, row_summary)
    residual_summary = summarize_residuals(row_summary)
    interval_summary, interval_rows = empirical_prediction_intervals(row_summary, pred)
    supplement = build_row_level_supplement(filtered, row_summary)

    return {
        "pred": pred,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "row_summary": row_summary,
        "range_summary": range_summary,
        "residual_summary": residual_summary,
        "prediction_interval_summary": interval_summary,
        "prediction_interval_rows": interval_rows,
        "supplement": supplement,
    }


def summarize_ranges(pred: pd.DataFrame, row_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario_id, model), group in row_summary.groupby(["scenario_id", "model"], dropna=False):
        qs = group["observed"].quantile([1 / 3, 2 / 3]).to_numpy()
        labels = ["low_observed_BMP", "mid_observed_BMP", "high_observed_BMP"]
        ranges = pd.cut(group["observed"], bins=[-np.inf, qs[0], qs[1], np.inf], labels=labels, include_lowest=True)
        tmp = group.copy()
        tmp["BMP_range"] = ranges.astype(str)
        for label, sub in tmp.groupby("BMP_range", dropna=False):
            metrics = metric_dict(sub["observed"], sub["mean_oof_prediction"])
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "model": model,
                    "BMP_range": label,
                    "range_min_observed": float(sub["observed"].min()),
                    "range_max_observed": float(sub["observed"].max()),
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "nested_cv_error_by_bmp_range.csv", index=False)
    out.to_csv(VALIDATED_DIR / "Table_validated_error_by_bmp_range.csv", index=False)
    return out


def summarize_residuals(row_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario_id, model), group in row_summary.groupby(["scenario_id", "model"], dropna=False):
        resid = group["residual_obs_minus_mean_oof"].dropna()
        rows.append(
            {
                "scenario_id": scenario_id,
                "model": model,
                "n": int(len(resid)),
                "mean_residual_obs_minus_pred": float(resid.mean()),
                "sd_residual": float(resid.std(ddof=1)),
                "q025": float(resid.quantile(0.025)),
                "q05": float(resid.quantile(0.05)),
                "median": float(resid.quantile(0.5)),
                "q95": float(resid.quantile(0.95)),
                "q975": float(resid.quantile(0.975)),
                "max_abs_residual": float(resid.abs().max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "nested_cv_residual_distribution_summary.csv", index=False)
    return out


def empirical_prediction_intervals(row_summary: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows = []
    summary_rows = []
    for (scenario_id, model), pooled in pred.groupby(["scenario_id", "model"], dropna=False):
        rows = row_summary[(row_summary["scenario_id"] == scenario_id) & (row_summary["model"] == model)].copy()
        residuals = pooled["observed"] - pooled["predicted"]
        for level, lo_q, hi_q in [(0.90, 0.05, 0.95), (0.95, 0.025, 0.975)]:
            lo_resid = float(residuals.quantile(lo_q))
            hi_resid = float(residuals.quantile(hi_q))
            tmp = rows.copy()
            tmp["interval_level"] = level
            tmp["lower_empirical_oof_residual_interval"] = tmp["mean_oof_prediction"] + lo_resid
            tmp["upper_empirical_oof_residual_interval"] = tmp["mean_oof_prediction"] + hi_resid
            tmp["covered"] = (
                (tmp["observed"] >= tmp["lower_empirical_oof_residual_interval"])
                & (tmp["observed"] <= tmp["upper_empirical_oof_residual_interval"])
            )
            tmp["interval_width"] = (
                tmp["upper_empirical_oof_residual_interval"] - tmp["lower_empirical_oof_residual_interval"]
            )
            interval_rows.extend(tmp.to_dict("records"))
            summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "model": model,
                    "interval_level": level,
                    "basis": "Empirical pooled repeated-CV out-of-fold residuals; not externally calibrated.",
                    "lower_residual_quantile_obs_minus_pred": lo_resid,
                    "upper_residual_quantile_obs_minus_pred": hi_resid,
                    "mean_interval_width": float(tmp["interval_width"].mean()),
                    "empirical_row_coverage": float(tmp["covered"].mean()),
                    "n_rows": int(len(tmp)),
                    "n_pooled_residuals": int(len(residuals)),
                }
            )
    interval_df = pd.DataFrame(interval_rows)
    summary_df = pd.DataFrame(summary_rows)
    interval_df.to_csv(PRED_DIR / "empirical_oof_prediction_intervals_by_row.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "empirical_oof_prediction_interval_summary.csv", index=False)
    summary_df.to_csv(VALIDATED_DIR / "Table_validated_prediction_interval_summary.csv", index=False)
    return summary_df, interval_df


def build_row_level_supplement(
    filtered: pd.DataFrame, row_summary: pd.DataFrame, apparent: pd.DataFrame | None = None
) -> pd.DataFrame:
    rf = row_summary[row_summary["model"] == "RF_nested_gridsearch"].copy()
    wide_parts = []
    for scenario_id, group in rf.groupby("scenario_id"):
        keep = group[
            [
                "source_row_id",
                "mean_oof_prediction",
                "sd_oof_prediction",
                "residual_obs_minus_mean_oof",
                "abs_error_mean_oof",
                "abs_pct_error_mean_oof",
            ]
        ].copy()
        keep = keep.rename(
            columns={
                "mean_oof_prediction": f"{scenario_id}__nested_oof_prediction_mean",
                "sd_oof_prediction": f"{scenario_id}__nested_oof_prediction_sd",
                "residual_obs_minus_mean_oof": f"{scenario_id}__nested_oof_residual_obs_minus_pred",
                "abs_error_mean_oof": f"{scenario_id}__nested_oof_abs_error",
                "abs_pct_error_mean_oof": f"{scenario_id}__nested_oof_abs_pct_error",
            }
        )
        wide_parts.append(keep)

    base_cols = [
        "source_row_id",
        "source_label",
        "Family",
        "Type",
        "Sub Type",
        RAW_DM,
        RAW_VS,
        RAW_RATIO,
        RAW_BMP_VS,
        "BMP_DM",
        "included_DM_ge_15",
        "complete_model_inputs",
    ] + FEATURES
    supplement = filtered[base_cols].drop_duplicates("source_row_id").copy()
    supplement = supplement.rename(columns={RAW_BMP_VS: "observed_BMP_VS", "BMP_DM": "computed_observed_BMP_DM"})
    for part in wide_parts:
        supplement = supplement.merge(part, on="source_row_id", how="left")
    if apparent is not None and not apparent.empty:
        apparent_wide_parts = []
        for scenario_id, group in apparent.groupby("scenario_id"):
            keep = group[["source_row_id", "apparent_full_data_prediction", "apparent_residual_obs_minus_pred"]].copy()
            keep = keep.rename(
                columns={
                    "apparent_full_data_prediction": f"{scenario_id}__apparent_full_data_prediction",
                    "apparent_residual_obs_minus_pred": f"{scenario_id}__apparent_full_data_residual_obs_minus_pred",
                }
            )
            apparent_wide_parts.append(keep)
        for part in apparent_wide_parts:
            supplement = supplement.merge(part, on="source_row_id", how="left")
    supplement["prediction_note"] = (
        "Nested-CV columns are out-of-fold repeated-CV predictions. Apparent full-data columns are fitted on all eligible rows and are not validation predictions."
    )
    supplement.to_csv(PRED_DIR / "final_row_level_supplementary_dataset_design.csv", index=False)

    design_lines = [
        "# Final Row-Level Supplementary Dataset Design",
        "",
        "The CSV design is saved at `analysis_outputs/predictions/final_row_level_supplementary_dataset_design.csv`.",
        "",
        "Required row identifiers and provenance columns:",
        "- `source_row_id`: stable row number from the available 131-row analytical table.",
        "- `source_label`, `Family`, `Type`, `Sub Type`: source identifiers retained from the Lallement-derived table.",
        "- `included_DM_ge_15`, `complete_model_inputs`: exclusion and eligibility flags.",
        "",
        "Observed targets and features:",
        "- `observed_BMP_VS`: observed BMP per VS from the source table.",
        "- `computed_observed_BMP_DM`: computed as BMP_VS x VS/DM.",
        "- All feature columns use source units in their column names.",
        "",
        "Prediction fields:",
        "- `*_nested_oof_prediction_mean`: mean of repeated nested-CV out-of-fold predictions for the row.",
        "- `*_nested_oof_prediction_sd`: between-repeat/fold prediction variability for the row.",
        "- `*_nested_oof_residual_obs_minus_pred`: observed minus mean out-of-fold prediction.",
        "- `*_nested_oof_abs_error` and `*_nested_oof_abs_pct_error`: row-level error summaries.",
        "",
        "Full-data fitted predictions, when supplied, must be stored separately and labeled `apparent`; they are not validation predictions.",
    ]
    (ROOT / "FINAL_ROW_LEVEL_SUPPLEMENTARY_DATASET_DESIGN.md").write_text("\n".join(design_lines) + "\n", encoding="utf-8")
    return supplement


def run_apparent_full_data_predictions(filtered: pd.DataFrame, scenarios: list[Scenario], grid_jobs: int) -> pd.DataFrame:
    rows = []
    param_rows = []
    for scenario in scenarios:
        X = filtered[scenario.feature_cols].copy()
        y = filtered[scenario.target_col].astype(float).copy()
        pipe, params, best_r2 = fit_full_grid_rf(X, y, grid_jobs)
        pred = pipe.predict(X)
        param_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "model": "RF_full_data_gridsearch_apparent",
                "basis": "apparent_full_data_refit_after_5fold_gridsearch",
                "best_inner_r2": best_r2,
                **params,
            }
        )
        for i, row in filtered.iterrows():
            obs = float(y.iloc[i])
            yhat = float(pred[i])
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "target_label": scenario.target_label,
                    "model": "RF_full_data_gridsearch_apparent",
                    "prediction_type": "apparent_full_data_fitted_not_validation",
                    "source_row_id": int(row["source_row_id"]),
                    "row_index_filtered": int(i),
                    "source_label": row["source_label"],
                    "Family": row["Family"],
                    "Type": row["Type"],
                    "Sub Type": row["Sub Type"],
                    "observed": obs,
                    "apparent_full_data_prediction": yhat,
                    "apparent_residual_obs_minus_pred": obs - yhat,
                    "abs_apparent_error": abs(obs - yhat),
                    "selected_params_json": json.dumps(params, sort_keys=True),
                }
            )
    out = pd.DataFrame(rows)
    params_out = pd.DataFrame(param_rows)
    out.to_csv(PRED_DIR / "apparent_full_data_predictions_rf.csv", index=False)
    params_out.to_csv(TABLE_DIR / "apparent_full_data_rf_gridsearch_params.csv", index=False)
    return out


def fit_full_grid_rf(
    X: pd.DataFrame, y: pd.Series, grid_jobs: int, optimized_grid: bool = True
) -> tuple[Pipeline, dict[str, Any], float]:
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
    if optimized_grid:
        search = manual_complete_rf_grid_search(X, y, inner, rf_jobs=grid_jobs)
    else:
        search = GridSearchCV(
            estimator=make_pipe(RandomForestRegressor(random_state=SEED, n_jobs=1)),
            param_grid=RF_GRID,
            scoring="r2",
            cv=inner,
            refit=True,
            n_jobs=grid_jobs,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X, y)
    return search.best_estimator_, normalize_grid_params(search.best_params_), float(search.best_score_)


def transformed_matrix(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    arr = pipe.named_steps["imputer"].transform(X)
    arr = pipe.named_steps["scaler"].transform(arr)
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def tree_shap_values(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if shap is None:
        raise RuntimeError("shap is not available")
    x_model = transformed_matrix(pipe, X)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    values = explainer.shap_values(x_model, check_additivity=False)
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values)


def feature_ranking_from_values(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": [clean(f) for f in features], "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp["rank"] = imp["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return imp.sort_values(["rank", "feature"]).reset_index(drop=True)


def pairwise_rank_agreement(rank_df: pd.DataFrame, basis_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = {k: g for k, g in rank_df.groupby(basis_cols + ["model"], dropna=False)}
    by_basis: dict[tuple[Any, ...], dict[str, pd.DataFrame]] = {}
    for key, group in grouped.items():
        basis = key[:-1]
        model = key[-1]
        by_basis.setdefault(basis, {})[model] = group
    for basis, model_map in by_basis.items():
        for m1, m2 in itertools.combinations(sorted(model_map), 2):
            left = model_map[m1][["feature", "rank"]].rename(columns={"rank": "rank_1"})
            right = model_map[m2][["feature", "rank"]].rename(columns={"rank": "rank_2"})
            merged = left.merge(right, on="feature", how="inner")
            spearman = stats.spearmanr(merged["rank_1"], merged["rank_2"]).correlation
            kendall = stats.kendalltau(merged["rank_1"], merged["rank_2"]).correlation
            top1 = set(model_map[m1].nsmallest(3, "rank")["feature"])
            top2 = set(model_map[m2].nsmallest(3, "rank")["feature"])
            row = {col: val for col, val in zip(basis_cols, basis)}
            row.update(
                {
                    "model_1": m1,
                    "model_2": m2,
                    "spearman_rank_correlation": float(spearman),
                    "kendall_rank_correlation": float(kendall),
                    "top3_overlap_count": int(len(top1.intersection(top2))),
                    "top3_overlap_features": "; ".join(sorted(top1.intersection(top2))),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def lignin_dependence_rows(
    values: np.ndarray,
    X_original: pd.DataFrame,
    scenario_id: str,
    model_name: str,
    basis: str,
    repeat: int | None = None,
    fold: int | None = None,
    dm_values_external: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if "Lignin Mean (g/100g DM)" not in X_original.columns:
        return []
    lignin_idx = list(X_original.columns).index("Lignin Mean (g/100g DM)")
    if dm_values_external is not None:
        dm_values = np.asarray(dm_values_external, dtype=float)
    elif RAW_DM in X_original.columns:
        dm_values = X_original[RAW_DM].to_numpy(dtype=float)
    else:
        dm_values = np.full(len(X_original), np.nan)
    rows = []
    for i, idx in enumerate(X_original.index):
        rows.append(
            {
                "scenario_id": scenario_id,
                "model": model_name,
                "basis": basis,
                "repeat": repeat,
                "fold": fold,
                "row_index_filtered": int(idx),
                "lignin": float(X_original.iloc[i]["Lignin Mean (g/100g DM)"]),
                "DM": float(dm_values[i]) if not np.isnan(dm_values[i]) else np.nan,
                "lignin_shap": float(values[i, lignin_idx]),
            }
        )
    return rows


def dependence_summary(dep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in dep.groupby(["scenario_id", "model", "basis"], dropna=False):
        scenario_id, model, basis = keys
        dm_median = group["DM"].median()
        for label, sub in {
            "all": group,
            "low_DM": group[group["DM"] <= dm_median],
            "high_DM": group[group["DM"] > dm_median],
        }.items():
            if len(sub) >= 3 and sub["lignin"].nunique() > 1:
                slope, intercept, r, p, se = stats.linregress(sub["lignin"], sub["lignin_shap"])
            else:
                slope = intercept = r = p = se = np.nan
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "model": model,
                    "basis": basis,
                    "stratum": label,
                    "n": int(len(sub)),
                    "dm_median_used": float(dm_median),
                    "slope_shap_per_lignin": float(slope),
                    "pearson_r_lignin_shap": float(r),
                    "p": float(p),
                    "slope_se": float(se),
                }
            )
    out = pd.DataFrame(rows)
    wide = out.pivot_table(
        index=["scenario_id", "model", "basis"],
        columns="stratum",
        values="slope_shap_per_lignin",
        aggfunc="first",
    ).reset_index()
    if {"low_DM", "high_DM"}.issubset(wide.columns):
        wide["low_minus_high_slope"] = wide["low_DM"] - wide["high_DM"]
        out = out.merge(wide[["scenario_id", "model", "basis", "low_minus_high_slope"]], on=["scenario_id", "model", "basis"], how="left")
    return out


def run_shap_analysis(
    filtered: pd.DataFrame,
    scenarios: list[Scenario],
    splits: list[dict[str, Any]],
    grid_jobs: int,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    shap_scenario_ids = {"A_BMP_VS_all_features", "B_BMP_DM_no_DM_VS", "C_BMP_DM_all_features"}
    scenarios = [s for s in scenarios if s.scenario_id in shap_scenario_ids]
    if shap is None:
        note = pd.DataFrame([{"status": "SHAP unavailable in the current Python environment."}])
        note.to_csv(SHAP_DIR / "shap_unavailable.csv", index=False)
        return {"unavailable": note}

    full_rank_rows: list[dict[str, Any]] = []
    heldout_rank_rows: list[dict[str, Any]] = []
    dependence_rows_all: list[dict[str, Any]] = []
    full_params_rows: list[dict[str, Any]] = []

    selected = pd.read_csv(TABLE_DIR / "nested_cv_selected_hyperparameters.csv")
    selected_map = {
        (r["scenario_id"], int(r["outer_id"])): {
            "n_estimators": int(r["n_estimators"]),
            "max_depth": parse_optional_depth(r["max_depth"]),
            "max_features": parse_max_features_value(r["max_features"]),
            "min_samples_leaf": int(r["min_samples_leaf"]),
            "min_samples_split": int(r["min_samples_split"]),
            "random_state": SEED,
            "n_jobs": 1,
        }
        for _, r in selected.iterrows()
    }

    for scenario in scenarios:
        X = filtered[scenario.feature_cols].copy()
        y = filtered[scenario.target_col].astype(float).copy()
        rf_pipe, rf_params, best_r2 = fit_full_grid_rf(X, y, grid_jobs)
        full_model_map = {
            "RandomForest": rf_pipe,
            "GradientBoosting": make_pipe(GradientBoostingRegressor(**GB_PARAMS)).fit(X, y),
            "ExtraTrees": make_pipe(ExtraTreesRegressor(**ET_PARAMS)).fit(X, y),
        }
        full_params_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "basis": "full_data_gridsearch_for_apparent_SHAP",
                "best_inner_r2": best_r2,
                **rf_params,
            }
        )
        for model_name, pipe in full_model_map.items():
            vals = tree_shap_values(pipe, X)
            ranking = feature_ranking_from_values(vals, scenario.feature_cols)
            for _, rr in ranking.iterrows():
                full_rank_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "model": model_name,
                        "basis": "full_data_apparent",
                        "feature": rr["feature"],
                        "mean_abs_shap": rr["mean_abs_shap"],
                        "rank": rr["rank"],
                    }
                )
            dependence_rows_all.extend(
                lignin_dependence_rows(
                    vals,
                    X,
                    scenario.scenario_id,
                    model_name,
                    "full_data_apparent",
                    dm_values_external=filtered[RAW_DM].to_numpy(dtype=float),
                )
            )

        for split in splits:
            X_train, X_test = X.iloc[split["train_idx"]], X.iloc[split["test_idx"]]
            y_train = y.iloc[split["train_idx"]]
            rf_params_fold = selected_map[(scenario.scenario_id, split["outer_id"])]
            heldout_models = shap_models_for_scenario(rf_params_fold)
            for model_name, pipe in heldout_models.items():
                fitted = clone(pipe).fit(X_train, y_train)
                vals = tree_shap_values(fitted, X_test)
                ranking = feature_ranking_from_values(vals, scenario.feature_cols)
                for _, rr in ranking.iterrows():
                    heldout_rank_rows.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "model": model_name,
                            "basis": "outer_fold_heldout",
                            "outer_id": split["outer_id"],
                            "repeat": split["repeat"],
                            "fold": split["fold"],
                            "feature": rr["feature"],
                            "mean_abs_shap": rr["mean_abs_shap"],
                            "rank": rr["rank"],
                        }
                    )
                dependence_rows_all.extend(
                    lignin_dependence_rows(
                        vals,
                        X_test,
                        scenario.scenario_id,
                        model_name,
                        "outer_fold_heldout",
                        split["repeat"],
                        split["fold"],
                        dm_values_external=filtered.iloc[split["test_idx"]][RAW_DM].to_numpy(dtype=float),
                    )
                )
        print(f"completed SHAP for {scenario.scenario_id}", flush=True)

    full_rank = pd.DataFrame(full_rank_rows)
    heldout_rank = pd.DataFrame(heldout_rank_rows)
    dep = pd.DataFrame(dependence_rows_all)
    full_params = pd.DataFrame(full_params_rows)

    full_agreement = pairwise_rank_agreement(full_rank, ["scenario_id", "basis"])
    heldout_agreement = pairwise_rank_agreement(heldout_rank, ["scenario_id", "basis", "outer_id", "repeat", "fold"])
    heldout_agreement_summary = (
        heldout_agreement.groupby(["scenario_id", "basis", "model_1", "model_2"], as_index=False)
        .agg(
            spearman_mean=("spearman_rank_correlation", "mean"),
            spearman_sd=("spearman_rank_correlation", "std"),
            kendall_mean=("kendall_rank_correlation", "mean"),
            kendall_sd=("kendall_rank_correlation", "std"),
            top3_overlap_mean=("top3_overlap_count", "mean"),
            top3_overlap_min=("top3_overlap_count", "min"),
        )
    )
    stability = (
        heldout_rank.groupby(["scenario_id", "model", "feature"], as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            sd_rank=("rank", "std"),
            median_rank=("rank", "median"),
            top1_rate=("rank", lambda s: float((s == 1).mean())),
            top3_rate=("rank", lambda s: float((s <= 3).mean())),
            mean_abs_shap=("mean_abs_shap", "mean"),
        )
        .sort_values(["scenario_id", "model", "mean_rank"])
    )
    dep_summary = dependence_summary(dep)

    full_rank.to_csv(SHAP_DIR / "full_data_shap_rankings_tree_models.csv", index=False)
    heldout_rank.to_csv(SHAP_DIR / "heldout_shap_fold_rankings_tree_models.csv", index=False)
    full_agreement.to_csv(SHAP_DIR / "full_data_shap_rank_agreement.csv", index=False)
    heldout_agreement.to_csv(SHAP_DIR / "heldout_shap_rank_agreement_by_fold.csv", index=False)
    heldout_agreement_summary.to_csv(SHAP_DIR / "heldout_shap_rank_agreement_summary.csv", index=False)
    stability.to_csv(SHAP_DIR / "heldout_shap_feature_rank_stability.csv", index=False)
    dep.to_csv(SHAP_DIR / "lignin_shap_dependence_values.csv", index=False)
    dep_summary.to_csv(SHAP_DIR / "lignin_shap_dependence_summary.csv", index=False)
    full_params.to_csv(SHAP_DIR / "full_data_rf_gridsearch_params_for_shap.csv", index=False)
    full_rank.to_csv(VALIDATED_DIR / "Table_validated_full_data_shap_rankings.csv", index=False)
    heldout_agreement_summary.to_csv(VALIDATED_DIR / "Table_validated_heldout_shap_rank_agreement_summary.csv", index=False)
    stability.to_csv(VALIDATED_DIR / "Table_validated_heldout_shap_rank_stability.csv", index=False)
    return {
        "full_rank": full_rank,
        "heldout_rank": heldout_rank,
        "full_agreement": full_agreement,
        "heldout_agreement_summary": heldout_agreement_summary,
        "stability": stability,
        "dependence": dep,
        "dependence_summary": dep_summary,
        "full_params": full_params,
    }


def run_lignin_dm_stats(filtered: pd.DataFrame, scenarios: list[Scenario]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for scenario in scenarios:
        X = filtered[scenario.feature_cols].copy()
        y = filtered[scenario.target_col].astype(float).to_numpy()
        lignin = filtered["Lignin Mean (g/100g DM)"].astype(float).to_numpy()
        dm = filtered[RAW_DM].astype(float).to_numpy()
        med = float(np.median(dm))
        low = dm <= med
        high = dm > med
        for label, mask in [("low_DM", low), ("high_DM", high), ("all", np.ones_like(low, dtype=bool))]:
            r, p = stats.pearsonr(lignin[mask], y[mask])
            slope, intercept, lr_r, slope_p, slope_se = stats.linregress(lignin[mask], y[mask])
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "test": "observed_lignin_target_association",
                    "stratum": label,
                    "n": int(mask.sum()),
                    "dm_median": med,
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "slope": float(slope),
                    "slope_p": float(slope_p),
                    "slope_se": float(slope_se),
                    "bootstrap_delta_r_ci_low": np.nan,
                    "bootstrap_delta_r_ci_high": np.nan,
                    "interaction_p": np.nan,
                    "note": "Observed association only; not a causal mechanism.",
                }
            )

        boot_delta = []
        for _ in range(5000):
            low_idx = rng.choice(np.where(low)[0], size=int(low.sum()), replace=True)
            high_idx = rng.choice(np.where(high)[0], size=int(high.sum()), replace=True)
            if len(np.unique(lignin[low_idx])) > 1 and len(np.unique(lignin[high_idx])) > 1:
                boot_delta.append(stats.pearsonr(lignin[low_idx], y[low_idx])[0] - stats.pearsonr(lignin[high_idx], y[high_idx])[0])
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "test": "bootstrap_delta_r_low_minus_high_DM",
                "stratum": "low_minus_high",
                "n": int(len(y)),
                "dm_median": med,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "slope": np.nan,
                "slope_p": np.nan,
                "slope_se": np.nan,
                "bootstrap_delta_r_ci_low": float(np.quantile(boot_delta, 0.025)),
                "bootstrap_delta_r_ci_high": float(np.quantile(boot_delta, 0.975)),
                "interaction_p": np.nan,
                "note": "Bootstrap CI for observed low-DM minus high-DM Pearson-r contrast.",
            }
        )

        # Simple median-DM interaction.
        high_float = high.astype(float)
        lignin_c = lignin - lignin.mean()
        x_simple = np.column_stack([np.ones(len(y)), lignin_c, high_float, lignin_c * high_float])
        p_simple = ols_p_value_for_last_term(x_simple, y)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "test": "median_split_lignin_x_high_DM_interaction",
                "stratum": "all",
                "n": int(len(y)),
                "dm_median": med,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "slope": np.nan,
                "slope_p": np.nan,
                "slope_se": np.nan,
                "bootstrap_delta_r_ci_low": np.nan,
                "bootstrap_delta_r_ci_high": np.nan,
                "interaction_p": float(p_simple),
                "note": "Unadjusted median-split slope-difference test.",
            }
        )

        # Adjusted continuous interaction using the scenario's feature set.
        if "Lignin Mean (g/100g DM)" in X.columns and RAW_DM in FEATURES:
            x_cols = []
            names = []
            for col in X.columns:
                arr = X[col].to_numpy(dtype=float)
                x_cols.append(arr - np.nanmean(arr))
                names.append(clean(col))
            interaction = (filtered["Lignin Mean (g/100g DM)"].to_numpy(dtype=float) - lignin.mean()) * (dm - dm.mean())
            x_adj = np.column_stack([np.ones(len(y)), *x_cols, interaction])
            p_adj = ols_p_value_for_last_term(x_adj, y)
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "test": "adjusted_continuous_lignin_x_DM_interaction",
                    "stratum": "all",
                    "n": int(len(y)),
                    "dm_median": med,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "slope": np.nan,
                    "slope_p": np.nan,
                    "slope_se": np.nan,
                    "bootstrap_delta_r_ci_low": np.nan,
                    "bootstrap_delta_r_ci_high": np.nan,
                    "interaction_p": float(p_adj),
                    "note": "Continuous interaction adjusted for the scenario feature set; interpret cautiously under collinearity.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "lignin_dm_hypothesis_sensitivity_analysis.csv", index=False)
    out.to_csv(VALIDATED_DIR / "Table_validated_lignin_dm_hypothesis_sensitivity.csv", index=False)
    return out


def ols_p_value_for_last_term(x: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    resid = y - pred
    n, p = x.shape
    df = n - p
    mse = float(np.sum(resid**2) / df)
    cov = mse * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    t_val = beta[-1] / se[-1]
    return float(2 * (1 - stats.t.cdf(abs(t_val), df)))


def make_plots(row_summary: pd.DataFrame, range_summary: pd.DataFrame) -> None:
    if plt is None or sns is None:
        return
    plot_models = ["RF_nested_gridsearch"]
    for scenario_id in sorted(row_summary["scenario_id"].unique()):
        sub = row_summary[(row_summary["scenario_id"] == scenario_id) & (row_summary["model"].isin(plot_models))]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        sns.scatterplot(data=sub, x="observed", y="mean_oof_prediction", hue="Family", ax=ax, s=45, edgecolor="0.3")
        lo = min(sub["observed"].min(), sub["mean_oof_prediction"].min())
        hi = max(sub["observed"].max(), sub["mean_oof_prediction"].max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
        ax.set_title(f"{scenario_id}: observed vs nested OOF prediction")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Mean nested OOF prediction")
        ax.legend(fontsize=7, frameon=False)

        ax = axes[1]
        sns.histplot(sub["residual_obs_minus_mean_oof"], kde=True, ax=ax, bins=18, color="#4c78a8")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_title("Residual distribution")
        ax.set_xlabel("Observed minus prediction")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{scenario_id}_rf_nested_observed_vs_pred_residuals.png", dpi=220)
        plt.close(fig)


def fmt(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(x)
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows available._"
    view = df[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: fmt(v, 3))
    return view.to_markdown(index=False)


def write_reports(
    meta: dict[str, Any],
    scenarios: list[Scenario],
    summaries: dict[str, pd.DataFrame],
    shap_outputs: dict[str, pd.DataFrame],
    lignin_stats: pd.DataFrame,
    n_repeats: int,
    n_splits: int,
) -> None:
    summary = summaries["summary"]
    range_summary = summaries["range_summary"]
    residual_summary = summaries["residual_summary"]
    interval_summary = summaries["prediction_interval_summary"]
    row_summary = summaries["row_summary"]
    selected = pd.read_csv(TABLE_DIR / "nested_cv_selected_hyperparameters.csv")
    selected_count = int(np.prod([len(v) for v in RF_GRID.values()]))

    primary_rf = summary[summary["model"] == "RF_nested_gridsearch"].copy()
    core_cols = [
        "scenario_id",
        "model",
        "R2_fold_mean",
        "R2_fold_ci_low",
        "R2_fold_ci_high",
        "RMSE_fold_mean",
        "MAE_fold_mean",
        "MAPE_pct_fold_mean",
        "nRMSE_mean_pct_fold_mean",
        "RPD_fold_mean",
        "bias_pred_minus_obs_fold_mean",
    ]

    nested_lines = [
        "# Nested CV Report",
        "",
        "## Scope",
        f"- Analytical table: `{DATA_PATH.name}`, SHA256 `{meta['sha256']}`.",
        f"- Raw rows available: {meta['raw_rows']}; rows after DM >= {DM_FILTER:g}% and complete inputs: {meta['filtered_rows']}.",
        "- Source-count reconciliation: Lallement et al. describe 132 feedstocks in narrative text, but the available analytical table contains 131 records; see `analysis_outputs/source_reconciliation/DATASET_PROVENANCE_RECONCILIATION.md`.",
        f"- Outer performance estimation: RepeatedKFold({n_splits} folds x {n_repeats} repeats, random_state=42).",
        "- Inner tuning: complete RF grid with KFold(5, shuffle=True, random_state=42), scoring=`r2`, refit=True. The default runner uses a prefix-forest optimization equivalent to evaluating the 50/100/150-tree candidates; `--sklearn-gridsearch` remains available as a slower fallback.",
        "- Preprocessing is inside every sklearn Pipeline: median imputation and StandardScaler are fit only on training folds.",
        "",
        "## Random Forest Search Space Used for Nested CV",
        f"- Candidate count per inner search: {selected_count}.",
        f"- `n_estimators`: {RF_GRID['model__n_estimators']}",
        f"- `max_depth`: {RF_GRID['model__max_depth']}",
        f"- `min_samples_split`: {RF_GRID['model__min_samples_split']}",
        f"- `min_samples_leaf`: {RF_GRID['model__min_samples_leaf']}",
        f"- `max_features`: {RF_GRID['model__max_features']}",
        "",
        "## RF Nested Performance By Target/Feature Scenario",
        markdown_table(primary_rf, core_cols),
        "",
        "## Fair Baseline Comparison",
        "All listed models use the same outer splits. Only Random Forest uses inner GridSearchCV; the remaining baselines are fixed-parameter comparisons.",
        markdown_table(
            summary.sort_values(["scenario_id", "R2_fold_mean"], ascending=[True, False]),
            ["scenario_id", "model", "R2_fold_mean", "RMSE_fold_mean", "MAE_fold_mean", "bias_pred_minus_obs_fold_mean"],
            max_rows=50,
        ),
        "",
        "## Residual And Range Checks",
        "- Residual summaries are saved in `analysis_outputs/tables/nested_cv_residual_distribution_summary.csv`.",
        "- Low/mid/high observed-BMP errors are saved in `analysis_outputs/tables/nested_cv_error_by_bmp_range.csv`.",
        "- Row-level repeated out-of-fold predictions are saved in `analysis_outputs/predictions/nested_cv_row_level_oof_summary.csv`.",
        "",
        "Key RF range errors:",
        markdown_table(
            range_summary[range_summary["model"] == "RF_nested_gridsearch"],
            ["scenario_id", "BMP_range", "n", "RMSE", "MAE", "bias_pred_minus_obs", "MAPE_pct"],
        ),
        "",
        "## Tuning Provenance",
        f"- Full nested candidate results are saved in `analysis_outputs/tables/nested_gridsearch_selected_results_by_outer.csv` ({selected_count} candidates per outer fold).",
        "- Selected hyperparameters per outer fold are saved in `analysis_outputs/tables/nested_cv_selected_hyperparameters.csv`.",
        "- Candidate-level tuning results are generated deterministically from the source table, documented grid, and fixed random seed.",
    ]
    (ROOT / "NESTED_CV_REPORT.md").write_text("\n".join(nested_lines) + "\n", encoding="utf-8")

    decision_rows = primary_rf[["scenario_id", "R2_fold_mean", "RMSE_fold_mean", "MAE_fold_mean", "bias_pred_minus_obs_fold_mean"]].copy()
    scenario_notes = pd.DataFrame(
        [{"scenario_id": s.scenario_id, "target": s.target_label, "interpretation_note": s.interpretation_note} for s in scenarios]
    )
    decision_rows = decision_rows.merge(scenario_notes, on="scenario_id", how="left")
    memo_lines = [
        "# Target And Feature Scenario Report",
        "",
        "## Scope",
        "This report summarizes the target and feature-set scenarios reproduced by the public workflow.",
        "",
        "## Scenario Evidence",
        markdown_table(decision_rows, ["scenario_id", "target", "R2_fold_mean", "RMSE_fold_mean", "MAE_fold_mean", "bias_pred_minus_obs_fold_mean", "interpretation_note"]),
        "",
        "## Coupling Assessment",
        "- `BMP_DM = BMP_VS x VS/DM` is exactly how the per-DM target is constructed in the available table.",
        "- Including DM and VS while predicting BMP per DM creates mathematical coupling because DM and VS reconstruct the conversion ratio used in the target.",
        "- Highest R2 is therefore not sufficient by itself to choose the primary scientific target.",
        "",
        "## Interpretation Boundary",
        "- The BMP per DM all-feature scenario should be interpreted with explicit target-coupling disclosure.",
        "- Scenario B is the conservative BMP per DM sensitivity because it excludes DM and VS predictors.",
        "- BMP per VS is a separate target specification normalized to volatile solids.",
        "",
        "## Interpretation Variables",
        "- Target specification determines the interpretation of model coefficients and feature-attribution outputs.",
        "- DM and VS membership differs across the main predictor set and sensitivity specifications.",
        "- Nested-CV RF estimates and fixed ensemble comparator results are separate result classes.",
    ]
    (ROOT / "TARGET_SENSITIVITY_REPORT.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")

    if "full_rank" in shap_outputs:
        full_rank = shap_outputs["full_rank"]
        full_agree = shap_outputs["full_agreement"]
        stability = shap_outputs["stability"]
        dep_summary = shap_outputs["dependence_summary"]
        lignin_stability = stability[stability["feature"] == "Lignin"].copy()
        shap_lines = [
            "# SHAP Cross-Model Report",
            "",
            "## Scope",
            "SHAP is evaluated as explanation stability, not as predictive or experimental validation. Full-data SHAP is apparent; held-out SHAP uses outer-fold test rows.",
            "",
            "## Full-Data Tree-Ensemble Rankings",
            markdown_table(full_rank[full_rank["rank"] <= 5], ["scenario_id", "model", "basis", "feature", "mean_abs_shap", "rank"], max_rows=60),
            "",
            "## Cross-Model Rank Agreement",
            markdown_table(full_agree, ["scenario_id", "model_1", "model_2", "spearman_rank_correlation", "kendall_rank_correlation", "top3_overlap_count", "top3_overlap_features"]),
            "",
            "## Held-Out Feature Rank Stability For Lignin",
            markdown_table(lignin_stability, ["scenario_id", "model", "feature", "mean_rank", "sd_rank", "top1_rate", "top3_rate", "mean_abs_shap"]),
            "",
            "## Lignin-DM Dependence Pattern",
            markdown_table(dep_summary, ["scenario_id", "model", "basis", "stratum", "n", "slope_shap_per_lignin", "low_minus_high_slope"], max_rows=80),
            "",
            "## Interpretation Boundary",
            "Lignin is a leading predictor across RF, Gradient Boosting and Extra Trees for the evaluated scenarios. The exact ordering of oxygen, hydrogen and DM is model-sensitive. The low-DM Lignin-SHAP slope contrast is exploratory and model-derived.",
        ]
    else:
        shap_lines = ["# SHAP Cross-Model Report", "", "SHAP could not be run in the current Python environment."]
    (ROOT / "SHAP_CROSS_MODEL_REPORT.md").write_text("\n".join(shap_lines) + "\n", encoding="utf-8")

    interval_lines = [
        "# Prediction Interval Report",
        "",
        "## Interval Definition",
        "Intervals are empirical intervals formed from pooled repeated-CV out-of-fold residual quantiles and centered on each row's mean repeated out-of-fold prediction. They are diagnostic uncertainty bands, not externally calibrated deployment intervals.",
        "",
        "## Empirical Coverage",
        markdown_table(interval_summary, ["scenario_id", "model", "interval_level", "mean_interval_width", "empirical_row_coverage", "n_rows", "n_pooled_residuals"], max_rows=100),
        "",
        "## Residual Distribution",
        markdown_table(residual_summary[residual_summary["model"] == "RF_nested_gridsearch"], ["scenario_id", "model", "mean_residual_obs_minus_pred", "sd_residual", "q025", "median", "q975", "max_abs_residual"]),
        "",
        "## Prediction-Error Scope",
        "These intervals summarize the spread of internal prediction errors. They are not externally calibrated individual prediction intervals and do not establish deployment performance without external calibration or a prespecified conformal validation design.",
    ]
    (ROOT / "PREDICTION_INTERVAL_REPORT.md").write_text("\n".join(interval_lines) + "\n", encoding="utf-8")

    lignin_table = (
        markdown_table(lignin_stats, ["scenario_id", "test", "stratum", "n", "pearson_r", "slope", "bootstrap_delta_r_ci_low", "bootstrap_delta_r_ci_high", "interaction_p", "note"], max_rows=80)
        if not lignin_stats.empty
        else "_Lignin-DM statistics were not run in this invocation._"
    )
    lignin_lines = [
        "# Lignin-DM Hypothesis Scope",
        "",
        "This report summarizes statistical context for the Lignin-DM pattern.",
        "",
        lignin_table,
    ]
    (OUT / "LIGNIN_DM_HYPOTHESIS_SCOPE.md").write_text("\n".join(lignin_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-jobs", type=int, default=min(max((os.cpu_count() or 2) - 1, 1), 8))
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-outer", type=int, default=None)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sklearn-gridsearch", action="store_true")
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--only-nested", action="store_true")
    parser.add_argument("--skip-lignin-stats", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    raw, filtered, meta = load_data()
    scenarios = scenario_by_id(build_scenarios(), args.scenarios)
    splits = build_outer_splits(len(filtered), args.n_repeats, args.n_splits, args.max_outer)
    save_fold_assignments(filtered, splits)

    if not args.skip_nested:
        run_nested_cv(
            filtered,
            scenarios,
            splits,
            args.grid_jobs,
            optimized_grid=not args.sklearn_gridsearch,
            force=args.force,
        )
    if args.only_nested:
        print("Nested CV smoke-test stage complete.")
        return

    summaries = summarize_predictions(filtered)
    make_plots(summaries["row_summary"], summaries["range_summary"])
    apparent = run_apparent_full_data_predictions(filtered, scenarios, args.grid_jobs)
    summaries["supplement"] = build_row_level_supplement(filtered, summaries["row_summary"], apparent)
    lignin_stats = pd.DataFrame()
    if not args.skip_lignin_stats:
        lignin_stats = run_lignin_dm_stats(filtered, scenarios)
    shap_outputs: dict[str, pd.DataFrame] = {}
    if not args.skip_shap:
        shap_outputs = run_shap_analysis(filtered, scenarios, splits, args.grid_jobs, force=args.force)
    if not args.skip_reports:
        write_reports(meta, scenarios, summaries, shap_outputs, lignin_stats, args.n_repeats, args.n_splits)
    print("Public analysis complete.")


if __name__ == "__main__":
    main()
