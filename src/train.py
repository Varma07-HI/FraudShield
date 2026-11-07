"""Train, compare, calibrate and evaluate fraud models. Run: python -m src.train"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.calibration import PlattCalibrator
from src.data import TARGET, add_features, feature_columns, load_raw, time_split
from src.metrics import best_threshold, net_savings, summarize, threshold_table

SEED = 42
REVIEW_COST = 5.0          # cost of a human reviewing one alert (currency units)
ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"


def xgb(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED,
    )


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # ---- 1. Load, build features, split by time ----
    df = add_features(load_raw(ROOT / "data" / "creditcard.csv"))
    feats = feature_columns(df)
    train, val, test = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(part):>7} rows, {int(part[TARGET].sum()):>4} frauds "
              f"({part[TARGET].mean():.3%}), Time {part['Time'].min():.0f}-{part['Time'].max():.0f}")
    Xtr, ytr = train[feats], train[TARGET]
    Xva, yva = val[feats], val[TARGET]
    Xte, yte = test[feats], test[TARGET]
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    # ---- 2. Imbalance study (XGBoost; judged on VALIDATION only) ----
    variants = {
        "no_resampling": xgb(1.0),
        "class_weight": xgb(pos_weight),
        "smote_in_pipeline": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=SEED)),
            ("clf", xgb(1.0)),
        ]),
    }
    study = []
    for name, model in variants.items():
        model.fit(Xtr, ytr)
        s = summarize(yva, model.predict_proba(Xva)[:, 1])
        study.append({"variant": name, **s})
        print(f"imbalance study | {name:<18} PR-AUC {s['pr_auc']:.4f}")
    study_df = pd.DataFrame(study)
    best_variant = study_df.sort_values("pr_auc", ascending=False).iloc[0]["variant"]
    print("best imbalance strategy on validation:", best_variant)

    # ---- 3. Model comparison (same time-based splits for every model) ----
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": Pipeline([("scale", StandardScaler()),
                              ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "xgboost": variants[best_variant],
    }
    val_scores, test_scores, results = {}, {}, []
    for name, model in models.items():
        if name != "xgboost":          # xgboost variant was already fitted above
            model.fit(Xtr, ytr)
        val_scores[name] = model.predict_proba(Xva)[:, 1]
        test_scores[name] = model.predict_proba(Xte)[:, 1]
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr)  # labels unused
    val_scores["isolation_forest"] = -iso.score_samples(Xva)
    test_scores["isolation_forest"] = -iso.score_samples(Xte)
    for name in val_scores:
        results.append({"model": name,
                        **{f"val_{k}": v for k, v in summarize(yva, val_scores[name]).items()},
                        **{f"test_{k}": v for k, v in summarize(yte, test_scores[name]).items()}})
    results_df = pd.DataFrame(results)
    print(results_df.round(4).to_string(index=False))

    # PR curves for the dashboard (downsampled)
    curves = []
    for name, sc in test_scores.items():
        p, r, _ = precision_recall_curve(yte, sc)
        idx = np.linspace(0, len(p) - 1, min(300, len(p))).astype(int)
        curves.append(pd.DataFrame({"model": name, "precision": p[idx], "recall": r[idx]}))
    pd.concat(curves).to_csv(REPORTS / "pr_curves.csv", index=False)

    # ---- 4. Calibrate the deployed model (Platt scaling fitted on validation) ----
    final = models["xgboost"]
    calibrator = PlattCalibrator().fit(val_scores["xgboost"], yva)
    val_cal = calibrator.predict(val_scores["xgboost"])
    test_cal = calibrator.predict(test_scores["xgboost"])
    brier = {"raw": float(brier_score_loss(yte, test_scores["xgboost"])),
             "calibrated": float(brier_score_loss(yte, test_cal))}
    print("Brier score on test (lower is better):", brier)
    frac_pos, mean_pred = calibration_curve(yte, test_cal, n_bins=10, strategy="quantile")
    pd.DataFrame({"mean_predicted": mean_pred, "fraction_fraud": frac_pos}).to_csv(
        REPORTS / "calibration.csv", index=False)

    # ---- 5. Pick the alert threshold from money, on VALIDATION; report on TEST ----
    val_table = threshold_table(yva, val_cal, val["Amount"])
    threshold = best_threshold(val_table, REVIEW_COST)
    test_table = threshold_table(yte, test_cal, test["Amount"])
    test_table.to_csv(REPORTS / "threshold_table.csv", index=False)
    flagged = test_cal >= threshold
    at_threshold = {
        "threshold": threshold, "review_cost": REVIEW_COST,
        "alerts": int(flagged.sum()),
        "frauds_caught": int((flagged & (yte == 1)).sum()),
        "frauds_missed": int(((~flagged) & (yte == 1)).sum()),
        "net_savings": float(
            test.loc[flagged & (yte == 1), "Amount"].sum() - REVIEW_COST * flagged.sum()),
        "total_fraud_amount": float(test.loc[yte == 1, "Amount"].sum()),
    }
    print("on the test set at the chosen threshold:", at_threshold)

    # ---- 6. SHAP explanations (the deployed model is always an XGBoost) ----
    core = final.named_steps["clf"] if hasattr(final, "named_steps") else final
    explainer = shap.TreeExplainer(core)
    rng = np.random.default_rng(SEED)
    explain_idx = rng.choice(len(Xte), size=min(2000, len(Xte)), replace=False)
    sv = explainer.shap_values(Xte.iloc[explain_idx])
    pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False).to_csv(REPORTS / "shap_importance.csv", index=False)

    # ---- 7. Save artefacts for the dashboard ----
    scored = test.copy()
    scored["score"] = test_cal
    legit = scored[scored[TARGET] == 0]
    sample = pd.concat([scored[scored[TARGET] == 1],
                        legit.sample(n=min(1500, len(legit)), random_state=SEED)]).sort_values("Time")
    sample.to_csv(REPORTS / "sample_transactions.csv", index=False)
    scored.sort_values("score", ascending=False).head(200).to_csv(REPORTS / "top_alerts.csv", index=False)
    hourly = df.assign(hour_index=(df["Time"] // 3600).astype(int)).groupby("hour_index").agg(
        transactions=(TARGET, "size"), frauds=(TARGET, "sum")).reset_index()
    hourly.to_csv(REPORTS / "hourly.csv", index=False)

    core.save_model(MODELS / "xgb_model.json")   # JSON survives library upgrades
    joblib.dump(calibrator, MODELS / "calibrator.joblib")
    config = {"features": feats, "threshold": threshold, "review_cost": REVIEW_COST,
              "best_imbalance_strategy": best_variant,
              "test_days": float((test["Time"].max() - test["Time"].min()) / 86400)}
    (MODELS / "config.json").write_text(json.dumps(config, indent=2))
    (REPORTS / "metrics.json").write_text(json.dumps(
        {"models": results, "imbalance_study": study, "brier": brier,
         "at_threshold": at_threshold}, indent=2))
    print("saved models/ and reports/")


if __name__ == "__main__":
    main()