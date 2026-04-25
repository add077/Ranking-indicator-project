from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
from itertools import product
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modele import (
    WalkForwardRolling,
    _add_prediction_rank_columns,
    _align_model_input_frame,
    _compute_binary_classification_metrics,
    _ensure_binary_target,
    _ensure_required_columns,
    _get_preprocessed_feature_names,
    _positive_class_proba,
    _prepare_temporal_frame,
    _safe_validation_metrics,
    evaluate_top_k_hit_rate,
)
from preprocessing import prepare_logit_inputs


LOGGER = logging.getLogger(__name__)


def prepare_rf_inputs(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    include_indicator_name: bool = True,
    include_signal_family: bool = True,
    include_horizon: bool = True,
    include_asset: bool = False,
    include_signal_active: bool = False,
    filter_active_only: bool = True,
    drop_na_target: bool = True,
    extra_numeric_features: Optional[Sequence[str]] = None,
    extra_categorical_features: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Prepare les entrees du Random Forest en reemployant la logique du pipeline logit.

    Le dataset final, la target et les features conservees sont donc strictement
    alignes avec ceux utilises pour la regression logistique.
    """
    return prepare_logit_inputs(
        df,
        target_col=target_col,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        include_indicator_name=include_indicator_name,
        include_signal_family=include_signal_family,
        include_horizon=include_horizon,
        include_asset=include_asset,
        include_signal_active=include_signal_active,
        filter_active_only=filter_active_only,
        drop_na_target=drop_na_target,
        extra_numeric_features=extra_numeric_features,
        extra_categorical_features=extra_categorical_features,
    )


def build_random_forest_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    n_estimators: int = 400,
    max_depth: Optional[int] = 8,
    min_samples_leaf: int = 20,
    max_features: str | int | float = "sqrt",
    class_weight: Optional[str | Mapping[Any, float]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Construit un pipeline sklearn pour un Random Forest.

    Choix retenus :
    - numeriques : imputation mediane, pas de standardisation ;
    - categorielles : imputation constante puis One-Hot Encoding ;
    - modele : RandomForestClassifier avec regularisation raisonnable.
    """
    if n_estimators <= 0:
        raise ValueError("`n_estimators` doit etre strictement positif.")
    if max_depth is not None and int(max_depth) <= 0:
        raise ValueError("`max_depth` doit etre positif ou `None`.")
    if min_samples_leaf <= 0:
        raise ValueError("`min_samples_leaf` doit etre strictement positif.")

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: List[tuple[str, Any, Sequence[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, list(numeric_features)))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, list(categorical_features)))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    classifier = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        class_weight=class_weight,
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def fit_random_forest_classifier(
    train_df: pd.DataFrame,
    *,
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    n_estimators: int = 400,
    max_depth: Optional[int] = 8,
    min_samples_leaf: int = 20,
    max_features: str | int | float = "sqrt",
    class_weight: Optional[str | Mapping[Any, float]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    include_indicator_name: bool = True,
    include_signal_family: bool = True,
    include_horizon: bool = True,
    include_asset: bool = False,
    filter_active_only: bool = True,
) -> Dict[str, Any]:
    """
    Fit un Random Forest sur le bloc train uniquement.
    """
    inputs = prepare_rf_inputs(
        train_df,
        target_col=target_col,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        include_indicator_name=include_indicator_name,
        include_signal_family=include_signal_family,
        include_horizon=include_horizon,
        include_asset=include_asset,
        include_signal_active=False,
        filter_active_only=filter_active_only,
        drop_na_target=True,
    )
    y_train = _ensure_binary_target(inputs["y"])

    pipeline = build_random_forest_pipeline(
        inputs["numeric_features"],
        inputs["categorical_features"],
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    pipeline.fit(inputs["X_model"], y_train)

    return {
        "pipeline": pipeline,
        "model_type": "random_forest_classifier",
        "n_estimators": int(n_estimators),
        "max_depth": None if max_depth is None else int(max_depth),
        "min_samples_leaf": int(min_samples_leaf),
        "max_features": max_features,
        "class_weight": class_weight,
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "numeric_features": inputs["numeric_features"],
        "categorical_features": inputs["categorical_features"],
        "feature_columns": inputs["feature_columns"],
        "model_input_columns": inputs["model_input_columns"],
        "context_feature_columns": inputs["context_feature_columns"],
        "df_train": inputs["df_model"],
        "X_train": inputs["X"],
        "y_train": y_train,
        "target_col": target_col,
        "datetime_col": datetime_col,
        "asset_col": asset_col,
        "indicator_col": indicator_col,
        "family_col": family_col,
        "horizon_col": horizon_col,
        "include_indicator_name": include_indicator_name,
        "include_signal_family": include_signal_family,
        "include_horizon": include_horizon,
        "include_asset": include_asset,
        "filter_active_only": filter_active_only,
        "n_train_obs": int(len(y_train)),
        "target_rate_train": float(y_train.mean()),
    }


def score_random_forest_classifier(
    model_bundle: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    filter_active_only: bool = False,
    add_rank: bool = True,
    rank_group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Score un dataset avec un Random Forest deja fitte.
    """
    pipeline = model_bundle["pipeline"]
    inputs = prepare_rf_inputs(
        df,
        target_col=target_col,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        include_indicator_name=bool(model_bundle.get("include_indicator_name", True)),
        include_signal_family=bool(model_bundle.get("include_signal_family", True)),
        include_horizon=bool(model_bundle.get("include_horizon", True)),
        include_asset=bool(model_bundle.get("include_asset", False)),
        include_signal_active=False,
        filter_active_only=filter_active_only,
        drop_na_target=False,
    )
    X_score = _align_model_input_frame(
        inputs["df_model"],
        model_bundle["model_input_columns"],
    )

    scored = inputs["df_model"].copy()
    proba_success = _positive_class_proba(pipeline, X_score)
    scored["pred_proba_success"] = proba_success
    clipped_proba = np.clip(proba_success, 1e-12, 1.0 - 1e-12)
    scored["pred_log_odds"] = np.log(clipped_proba / (1.0 - clipped_proba))
    scored["pred_class"] = (proba_success >= 0.5).astype(int)

    if add_rank:
        if rank_group_cols is None:
            rank_group_cols = [datetime_col, asset_col, horizon_col]
        scored = _add_prediction_rank_columns(
            scored,
            prob_col="pred_proba_success",
            rank_group_cols=rank_group_cols,
        )

    return scored


def train_validate_random_forest_grid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_estimators_grid: Sequence[int] = (300, 500),
    max_depth_grid: Sequence[Optional[int]] = (6, 8, 12),
    min_samples_leaf_grid: Sequence[int] = (10, 20, 50),
    max_features_grid: Sequence[str | float | int] = ("sqrt",),
    class_weight_grid: Sequence[Optional[str | Mapping[Any, float]]] = (None, "balanced"),
    target_col: str = "target",
    datetime_col: str = "datetime",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    filter_active_only: bool = True,
) -> pd.DataFrame:
    """
    Compare quelques configurations Random Forest sur un bloc d'evaluation.
    """
    rows: List[Dict[str, Any]] = []

    for n_estimators, max_depth, min_samples_leaf, max_features, class_weight in product(
        n_estimators_grid,
        max_depth_grid,
        min_samples_leaf_grid,
        max_features_grid,
        class_weight_grid,
    ):
        row: Dict[str, Any] = {
            "n_estimators": int(n_estimators),
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
            "class_weight": class_weight,
            "fit_ok": False,
            "error": None,
        }

        try:
            bundle = fit_random_forest_classifier(
                train_df,
                target_col=target_col,
                datetime_col=datetime_col,
                asset_col=asset_col,
                indicator_col=indicator_col,
                family_col=family_col,
                horizon_col=horizon_col,
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                min_samples_leaf=int(min_samples_leaf),
                max_features=max_features,
                class_weight=class_weight,
                filter_active_only=filter_active_only,
            )
            scored_eval = score_random_forest_classifier(
                bundle,
                test_df,
                target_col=target_col,
                datetime_col=datetime_col,
                asset_col=asset_col,
                indicator_col=indicator_col,
                family_col=family_col,
                horizon_col=horizon_col,
                filter_active_only=filter_active_only,
                add_rank=False,
            )
            prob_metrics = _safe_validation_metrics(
                scored_eval[target_col],
                scored_eval["pred_proba_success"],
            )
            class_metrics = _compute_binary_classification_metrics(
                pd.to_numeric(scored_eval[target_col], errors="coerce"),
                pd.to_numeric(scored_eval["pred_class"], errors="coerce"),
                pd.to_numeric(scored_eval["pred_proba_success"], errors="coerce"),
            )

            row.update(prob_metrics)
            row["accuracy"] = class_metrics["accuracy"]
            row["f1"] = class_metrics["f1"]
            row["fit_ok"] = True
            row["n_train_obs"] = bundle["n_train_obs"]
            row["target_rate_train"] = bundle["target_rate_train"]
        except Exception as exc:
            LOGGER.warning(
                "Configuration RF ignoree (n_estimators=%s, max_depth=%s, leaf=%s, max_features=%s, class_weight=%s) : %s",
                n_estimators,
                max_depth,
                min_samples_leaf,
                max_features,
                class_weight,
                exc,
            )
            row["error"] = str(exc)
            row.update(
                {
                    "n_obs": np.nan,
                    "target_rate": np.nan,
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "roc_auc": np.nan,
                    "log_loss": np.nan,
                    "brier_score": np.nan,
                    "n_train_obs": np.nan,
                    "target_rate_train": np.nan,
                }
            )

        rows.append(row)

    results = pd.DataFrame(rows)
    return results.sort_values(
        ["fit_ok", "log_loss", "brier_score", "roc_auc"],
        ascending=[False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def _clean_preprocessed_feature_name(
    feature_name: str,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[str, str, Optional[str]]:
    if feature_name.startswith("numeric__"):
        clean = feature_name.replace("numeric__", "", 1)
        return clean, "numeric", clean

    if feature_name.startswith("categorical__"):
        clean = feature_name.replace("categorical__", "", 1)
        for source in sorted(categorical_features, key=len, reverse=True):
            prefix = f"{source}_"
            if clean.startswith(prefix):
                return clean, "categorical", source
        return clean, "categorical", None

    return feature_name, "unknown", None


def extract_random_forest_feature_importance(
    model_bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Extrait les importances de features du Random Forest apres preprocessing.
    """
    pipeline = model_bundle["pipeline"]
    classifier = pipeline.named_steps["model"]
    feature_names = _get_preprocessed_feature_names(pipeline)
    importances = np.asarray(classifier.feature_importances_).ravel()
    if len(feature_names) != len(importances):
        raise ValueError(
            "Nombre d'importances different du nombre de features preprocessees."
        )

    rows = []
    for feature_name, importance in zip(feature_names, importances):
        clean_name, feature_type, source_feature = _clean_preprocessed_feature_name(
            feature_name,
            numeric_features=model_bundle.get("numeric_features", []),
            categorical_features=model_bundle.get("categorical_features", []),
        )
        rows.append(
            {
                "feature_name": clean_name,
                "raw_feature_name": feature_name,
                "source_feature": source_feature,
                "feature_type": feature_type,
                "importance": float(importance),
            }
        )

    importance_df = pd.DataFrame(rows).sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)
    importance_df["importance_rank"] = np.arange(1, len(importance_df) + 1, dtype=int)
    return importance_df


@dataclass
class RollingWindowBacktester:
    """
    Backtester generique fold-par-fold pour classifieurs sklearn-like.

    Il reutilise :
    - `WalkForwardRolling` pour le split temporel / embargo / purge ;
    - les fonctions de fit / score fournies ;
    - les metriques de classification et le ranking du pipeline existant.
    """

    cv: WalkForwardRolling
    fit_function: Callable[..., Dict[str, Any]]
    score_function: Callable[..., pd.DataFrame]
    datetime_col: str = "date"
    asset_col: str = "asset"
    indicator_col: str = "indicator_name"
    family_col: str = "signal_family"
    horizon_col: str = "horizon"
    target_col: str = "target"
    fit_kwargs: Optional[Mapping[str, Any]] = None
    score_kwargs: Optional[Mapping[str, Any]] = None
    drop_na_target: bool = True
    verbose: int = 0

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        _ensure_required_columns(df, [self.datetime_col, self.horizon_col, self.target_col])
        prepared = _prepare_temporal_frame(
            df,
            datetime_col=self.datetime_col,
            horizon_col=self.horizon_col,
            target_col=self.target_col,
        )

        fit_kwargs = dict(self.fit_kwargs or {})
        score_kwargs = dict(self.score_kwargs or {})
        if "filter_active_only" in fit_kwargs and "filter_active_only" not in score_kwargs:
            score_kwargs["filter_active_only"] = fit_kwargs["filter_active_only"]

        fold_rows: List[Dict[str, Any]] = []
        fold_metrics_rows: List[Dict[str, Any]] = []
        oos_frames: List[pd.DataFrame] = []
        fitted_bundles: List[Dict[str, Any]] = []

        for fold_id, (train_idx, test_idx) in enumerate(self.cv.split(prepared), start=1):
            train_fold = prepared.iloc[train_idx].drop(columns="__row_order", errors="ignore").copy()
            test_fold = prepared.iloc[test_idx].drop(columns="__row_order", errors="ignore").copy()

            if train_fold.empty or test_fold.empty:
                LOGGER.warning("Fold %d ignore car train ou test est vide.", fold_id)
                continue

            try:
                bundle = self.fit_function(
                    train_fold,
                    target_col=self.target_col,
                    datetime_col=self.datetime_col,
                    asset_col=self.asset_col,
                    indicator_col=self.indicator_col,
                    family_col=self.family_col,
                    horizon_col=self.horizon_col,
                    **fit_kwargs,
                )
                fitted_bundles.append(bundle)

                scored_test = self.score_function(
                    bundle,
                    test_fold,
                    target_col=self.target_col,
                    datetime_col=self.datetime_col,
                    asset_col=self.asset_col,
                    indicator_col=self.indicator_col,
                    family_col=self.family_col,
                    horizon_col=self.horizon_col,
                    **score_kwargs,
                )
            except Exception as exc:
                LOGGER.warning("Fold %d ignore suite a une erreur de fit/score : %s", fold_id, exc)
                continue

            if self.drop_na_target:
                scored_test = scored_test.loc[
                    pd.to_numeric(scored_test[self.target_col], errors="coerce").notna()
                ].copy()
                if scored_test.empty:
                    LOGGER.warning("Fold %d ignore apres drop des targets manquantes.", fold_id)
                    continue

            scored_test["fold_id"] = fold_id
            scored_test["y_true"] = pd.to_numeric(scored_test[self.target_col], errors="coerce")
            scored_test["y_pred"] = pd.to_numeric(scored_test["pred_class"], errors="coerce")
            scored_test["y_proba_pred"] = pd.to_numeric(scored_test["pred_proba_success"], errors="coerce")

            fold_summary: Dict[str, Any] = {}
            if hasattr(self.cv, "fold_summaries_") and len(getattr(self.cv, "fold_summaries_", [])) >= fold_id:
                fold_summary = deepcopy(self.cv.fold_summaries_[fold_id - 1])
                for key, value in fold_summary.items():
                    scored_test[key] = value
            else:
                fold_summary = {
                    "fold_id": fold_id,
                    "train_start": pd.to_datetime(train_fold[self.datetime_col]).min(),
                    "train_end": pd.to_datetime(train_fold[self.datetime_col]).max(),
                    "test_start": pd.to_datetime(test_fold[self.datetime_col]).min(),
                    "test_end": pd.to_datetime(test_fold[self.datetime_col]).max(),
                }

            metrics = _compute_binary_classification_metrics(
                scored_test["y_true"],
                scored_test["y_pred"],
                scored_test["y_proba_pred"],
            )
            metrics_row = {
                **fold_summary,
                **metrics,
                "n_train_used_for_fit": int(bundle.get("n_train_obs", len(train_fold))),
                "n_test_scored": int(len(scored_test)),
            }
            fold_rows.append(fold_summary)
            fold_metrics_rows.append(metrics_row)
            oos_frames.append(scored_test)

            if self.verbose >= 1:
                LOGGER.info(
                    "Fold %d | accuracy=%.4f | f1=%.4f | auc=%s | log_loss=%s | n_test=%d",
                    fold_id,
                    metrics["accuracy"] if pd.notna(metrics["accuracy"]) else np.nan,
                    metrics["f1"] if pd.notna(metrics["f1"]) else np.nan,
                    f"{metrics['roc_auc']:.4f}" if pd.notna(metrics["roc_auc"]) else "nan",
                    f"{metrics['log_loss']:.4f}" if pd.notna(metrics["log_loss"]) else "nan",
                    int(metrics["n_obs"]) if pd.notna(metrics["n_obs"]) else 0,
                )

        if not oos_frames:
            raise ValueError(
                "Aucune prediction out-of-sample n'a ete produite. "
                "Verifie les periodes du rolling window, l'embargo et le filtrage."
            )

        oos_predictions_df = pd.concat(oos_frames, axis=0, ignore_index=True)
        rank_group_cols = [self.datetime_col, self.asset_col, self.horizon_col]
        rank_group_cols = [col for col in rank_group_cols if col in oos_predictions_df.columns]
        if "rank_pred_proba" not in oos_predictions_df.columns and rank_group_cols:
            oos_predictions_df = _add_prediction_rank_columns(
                oos_predictions_df,
                prob_col="pred_proba_success",
                rank_group_cols=rank_group_cols,
            )

        fold_metrics_df = pd.DataFrame(fold_metrics_rows).sort_values("fold_id").reset_index(drop=True)
        global_metrics = _compute_binary_classification_metrics(
            oos_predictions_df["y_true"],
            oos_predictions_df["y_pred"],
            oos_predictions_df["y_proba_pred"],
        )

        return {
            "cv": self.cv,
            "backtester": self,
            "fitted_bundles": fitted_bundles,
            "fold_summaries": pd.DataFrame(fold_rows).sort_values("fold_id").reset_index(drop=True),
            "fold_metrics_df": fold_metrics_df,
            "accuracy_scores": pd.Series(fold_metrics_df["accuracy"].values, index=fold_metrics_df["fold_id"]),
            "f1_scores": pd.Series(fold_metrics_df["f1"].values, index=fold_metrics_df["fold_id"]),
            "roc_auc_scores": pd.Series(fold_metrics_df["roc_auc"].values, index=fold_metrics_df["fold_id"]),
            "log_loss_scores": pd.Series(fold_metrics_df["log_loss"].values, index=fold_metrics_df["fold_id"]),
            "brier_scores": pd.Series(fold_metrics_df["brier_score"].values, index=fold_metrics_df["fold_id"]),
            "y_true": oos_predictions_df["y_true"].reset_index(drop=True),
            "y_pred": oos_predictions_df["y_pred"].reset_index(drop=True),
            "y_proba_pred": oos_predictions_df["y_proba_pred"].reset_index(drop=True),
            "global_metrics": global_metrics,
            "oos_predictions_df": oos_predictions_df,
        }


def run_rolling_window_backtest(
    df: pd.DataFrame,
    *,
    fit_function: Callable[..., Dict[str, Any]],
    score_function: Callable[..., pd.DataFrame],
    datetime_col: str = "date",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    target_col: str = "target",
    horizon_steps_map: Mapping[str, int],
    period_train: int,
    period_test: int,
    period_embargo: int,
    purge: bool = True,
    drop_na_target: bool = True,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    score_kwargs: Optional[Mapping[str, Any]] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Backtest rolling generique reutilisant le split `WalkForwardRolling`.
    """
    if not purge:
        raise ValueError(
            "Le backtest rolling de ce projet repose sur `WalkForwardRolling`, "
            "qui applique une purge horizon-aware. Utilise `purge=True`."
        )

    cv = WalkForwardRolling(
        period_train=period_train,
        period_test=period_test,
        period_embargo=period_embargo,
        datetime_col=datetime_col,
        horizon_col=horizon_col,
        horizon_steps_map=horizon_steps_map,
        verbose=verbose,
    )
    backtester = RollingWindowBacktester(
        cv=cv,
        fit_function=fit_function,
        score_function=score_function,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        target_col=target_col,
        fit_kwargs=fit_kwargs,
        score_kwargs=score_kwargs,
        drop_na_target=drop_na_target,
        verbose=verbose,
    )
    return backtester.run(df)


def run_random_forest_rolling_backtest(
    df: pd.DataFrame,
    *,
    datetime_col: str = "date",
    asset_col: str = "asset",
    indicator_col: str = "indicator_name",
    family_col: str = "signal_family",
    horizon_col: str = "horizon",
    target_col: str = "target",
    horizon_steps_map: Mapping[str, int],
    period_train: int,
    period_test: int,
    period_embargo: int,
    n_estimators: int = 400,
    max_depth: Optional[int] = 8,
    min_samples_leaf: int = 20,
    max_features: str | int | float = "sqrt",
    class_weight: Optional[str | Mapping[Any, float]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    include_indicator_name: bool = True,
    include_signal_family: bool = True,
    include_horizon: bool = True,
    include_asset: bool = False,
    filter_active_only: bool = True,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Wrapper pratique pour le benchmark RF sur le meme protocole rolling que la logistique.
    """
    return run_rolling_window_backtest(
        df,
        fit_function=fit_random_forest_classifier,
        score_function=score_random_forest_classifier,
        datetime_col=datetime_col,
        asset_col=asset_col,
        indicator_col=indicator_col,
        family_col=family_col,
        horizon_col=horizon_col,
        target_col=target_col,
        horizon_steps_map=horizon_steps_map,
        period_train=period_train,
        period_test=period_test,
        period_embargo=period_embargo,
        purge=True,
        drop_na_target=True,
        fit_kwargs={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "include_indicator_name": include_indicator_name,
            "include_signal_family": include_signal_family,
            "include_horizon": include_horizon,
            "include_asset": include_asset,
            "filter_active_only": filter_active_only,
        },
        score_kwargs={
            "filter_active_only": filter_active_only,
            "add_rank": True,
        },
        verbose=verbose,
    )


__all__ = [
    "WalkForwardRolling",
    "build_random_forest_pipeline",
    "evaluate_top_k_hit_rate",
    "extract_random_forest_feature_importance",
    "fit_random_forest_classifier",
    "prepare_rf_inputs",
    "RollingWindowBacktester",
    "run_random_forest_rolling_backtest",
    "run_rolling_window_backtest",
    "score_random_forest_classifier",
    "train_validate_random_forest_grid",
]
