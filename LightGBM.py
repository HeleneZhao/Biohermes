#Python (v3.11.13)
import argparse
import random
import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Global seed
# -------------------------------------------------------------------

SEED = 2025


def set_global_seed(seed: int = SEED):
    np.random.seed(seed)
    random.seed(seed)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def infer_labels_from_dap_path(dap_path: str):
    
    fname = Path(dap_path).name
    m_case = re.search(r"LG_(.+)_vs_", fname)
    m_ctrl = re.search(r"vs_(.+)_", fname)
    if not (m_case and m_ctrl):
        raise ValueError(f"Cannot infer case/control labels from filename: {fname}")
    case_label = m_case.group(1)
    control_label = m_ctrl.group(1)
    return (case_label,), (control_label,)


def normal_imp(mydict):
    """Normalize importance dictionary so values sum to 1."""
    mysum = float(sum(mydict.values()))
    if mysum == 0:
        return mydict
    for key in list(mydict.keys()):
        mydict[key] = mydict[key] / mysum
    return mydict


def make_unique_gene_symbols(protein_ids, soma_to_symbol):
    
    used = {}
    feature_names = []
    protein_to_symbol_unique = {}
    for p in protein_ids:
        base = soma_to_symbol.get(p, p)
        if base not in used:
            used[base] = 1
            name = base
        else:
            used[base] += 1
            name = f"{base}_{used[base]}"
        feature_names.append(name)
        protein_to_symbol_unique[p] = name
    return feature_names, protein_to_symbol_unique


def objective(trial, X, y, cv_splits):
    """Optuna objective for LightGBM hyperparameter tuning using fixed training folds."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": True,
        "verbosity": -1,
        "random_state": SEED,

        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10]),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),

        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
    }

    aucs = []
    for tr_idx, val_idx in cv_splits:
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
        )
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

    return float(np.mean(aucs))


def model_val(X, y, cv_splits, n_trials=100, seed=SEED):
    """Run Optuna using fixed CV splits on the training set."""
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X, y, cv_splits), n_trials=n_trials)

    print("  # finished trials:", len(study.trials))
    print("  Best params:", study.best_trial.params)
    print("  Best CV AUC:", study.best_trial.value)

    return study.best_trial.params


def get_imp_analy(Imp_df, top_prop):
    """Number of top proteins needed to accumulate 'top_prop' of TotalGain."""
    imp_score, it = 0.0, 0
    while imp_score < top_prop and it < len(Imp_df):
        imp_score += float(Imp_df.TotalGain_cv.iloc[it])
        it += 1
    return it + 1


def choose_k_by_auc_gain_plateau(
    sfs_df,
    delta: float = 0.005,
    patience: int = 2,
    auc_col: str = "AUC_all",
):
    """Choose k by detecting an AUC gain plateau."""
    auc = sfs_df[auc_col].astype(float).values
    if len(auc) == 0:
        return 1

    gains = np.diff(auc, prepend=auc[0])

    bad = 0
    k = 1
    for i in range(2, len(auc) + 1):  # i is 1-based k
        if gains[i - 1] < delta:
            bad += 1
        else:
            bad = 0

        k = i
        if bad >= patience:
            k = i - patience
            break

    return max(1, int(k))


# --- DeLong helpers (kept for reporting in sd_SFS.csv) ---

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


# -------------------------------------------------------------------
# Main pipeline for ONE (PCA, DAP) pair
# -------------------------------------------------------------------

def run_protein_selection_experiment(
    base_dir,
    pca_file,
    soma_annotation_file,
    dap_file,
    target_col,
    case_labels,
    control_labels,
    output_root,
    n_optuna_trials,
    top_prop,
    delong_col: int = 2,
    ts: str = "",
    plateau_delta: float = 0.005,
    plateau_patience: int = 2,
):
    """Run full LightGBM protein-selection pipeline for one PCA × one DAP file."""
    set_global_seed(SEED)

    base_dir = Path(base_dir)
    pca_path = base_dir / pca_file
    dap_path = base_dir / dap_file
    soma_path = base_dir / soma_annotation_file

    exp_name = f"{pca_path.stem}__{dap_path.stem}"
    result_dir = Path(f"{output_root}/{exp_name}")
    result_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "base_dir": str(base_dir),
        "pca_file": str(pca_file),
        "soma_annotation_file": str(soma_annotation_file),
        "dap_file": str(dap_file),
        "target_col": target_col,
        "case_labels": list(case_labels),
        "control_labels": list(control_labels),
        "n_optuna_trials": int(n_optuna_trials),
        "top_prop": float(top_prop),
        "seed": int(SEED),
        "delong_col": int(delong_col),
        "plateau_delta": float(plateau_delta),
        "plateau_patience": int(plateau_patience),
    }
    with open(result_dir / "run_args.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"PCA file: {pca_path}")
    print(f"DAPs file: {dap_path}")
    print(f"Output dir: {result_dir}")
    print("=" * 80)

    # ------------------ Step 1. Ranking DAPs ------------------ #
    print("Step 1: Ranking DAPs with LightGBM + SHAP")

    mydf = pd.read_csv(pca_path)

    dict_df = pd.read_csv(
        soma_path,
        usecols=["Protein", "UniProt", "EntrezGeneID", "EntrezGeneSymbol"],
    )
    soma_to_symbol = dict_df.set_index("Protein")["EntrezGeneSymbol"].to_dict()

    my_f_df = pd.read_csv(dap_path)
    my_f_df = my_f_df.loc[my_f_df["padjust_BH"] < 0.05].copy()
    my_f_df = my_f_df.drop_duplicates(subset="Protein", keep="first")
    my_f_lst = my_f_df.Protein.tolist()
    print(f"  # DAP proteins (padjust_BH < 0.05): {len(my_f_lst)}")

    if target_col not in mydf.columns:
        print(f"  [SKIP] target_col '{target_col}' not found in PCA file.")
        return

    mydf["target_y"] = mydf[target_col].copy()
    mydf = mydf.loc[mydf["target_y"].isin(case_labels + control_labels)]
    mydf.reset_index(inplace=True, drop=True)

    if mydf.shape[0] == 0:
        print("  [SKIP] No samples after label filtering.")
        return

    mydf["target_y"].replace(case_labels + control_labels, [1, 0], inplace=True)

    if mydf["target_y"].nunique() < 2:
        print("  [SKIP] Only one class present after filtering.")
        return

    my_X_all = mydf[my_f_lst]
    y_all = mydf["target_y"]
    n_samples = len(y_all)

    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.3,
        stratify=y_all,
        random_state=SEED,
    )

    np.save(result_dir / "train_idx.npy", train_idx)
    np.save(result_dir / "test_idx.npy", test_idx)

    X_train = my_X_all.iloc[train_idx].reset_index(drop=True)
    y_train = y_all.iloc[train_idx].reset_index(drop=True)
    X_test = my_X_all.iloc[test_idx].reset_index(drop=True)
    y_test = y_all.iloc[test_idx].reset_index(drop=True)

    skf_train = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_splits_train = list(skf_train.split(X_train, y_train))
    joblib.dump(cv_splits_train, result_dir / "cv_splits_train.pkl")
    nb_folds_train = len(cv_splits_train)

    best_params = model_val(
        X_train,
        y_train,
        cv_splits=cv_splits_train,
        n_trials=n_optuna_trials,
        seed=SEED,
    )

    final_model = LGBMClassifier(
        **best_params, objective="binary", is_unbalance=True, random_state=SEED
    )
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, final_preds)
    print(f"  Hold-out test AUC (all DAPs): {test_auc:.4f}\n")

    best_params_path = result_dir / "best_params_step1.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    best_model_path = result_dir / "best_model_step1.pkl"
    joblib.dump(final_model, best_model_path)
    print(f"  Saved Step1 params to {best_params_path}")
    print(f"  Saved Step1 model to  {best_model_path}")

    y_test_bin = (final_preds >= 0.5).astype(int)
    test_metrics_step1 = {
        "roc_auc": float(test_auc),
        "accuracy_0.5": float(accuracy_score(y_test, y_test_bin)),
        "f1_0.5": float(f1_score(y_test, y_test_bin)),
        "precision_0.5": float(precision_score(y_test, y_test_bin)),
        "recall_0.5": float(recall_score(y_test, y_test_bin)),
        "n_test": int(len(y_test)),
    }
    with open(result_dir / "metrics_step1_test.json", "w") as f:
        json.dump(test_metrics_step1, f, indent=2)

    pred_df_step1 = pd.DataFrame(
        {"idx": test_idx, "y_true": y_test.values, "y_pred_prob": final_preds}
    )
    pred_df_step1.to_csv(result_dir / "predictions_step1_test.csv", index=False)
    print("  Saved Step1 metrics and predictions.\n")

    # ------------------ Step 1b. SHAP beeswarm for all DAP proteins (training only) ------------------ #
    print("Step 1b: SHAP beeswarm for all DAP proteins (training only)")

    # rename to unique gene symbols for plotting (model still uses original column names)
    feat_names_step1, _ = make_unique_gene_symbols(list(X_train.columns), soma_to_symbol)
    X_train_shap = X_train.copy()
    X_train_shap.columns = feat_names_step1

    TOP_K_SHAP_STEP1 = min(40, X_train_shap.shape[1])

    explainer_step1 = shap.Explainer(final_model, X_train_shap)
    shap_values_step1 = explainer_step1(X_train_shap)

    shap.plots.beeswarm(shap_values_step1, max_display=TOP_K_SHAP_STEP1)
    plt.gcf().set_size_inches(12, 6)
    ax = plt.gca()
    ax.set_ylabel("DAP proteins (all features, Step 1)", fontsize=18, weight="bold")
    ax.set_xlabel("SHAP values", fontsize=14, weight="bold")
    ax.tick_params(axis="x", labelsize=14)
    ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels, fontsize=10, color="black")
    plt.tight_layout()
    outimg_shap_step1 = result_dir / "sa_SHAP_AllDAPs_train.png"
    plt.savefig(outimg_shap_step1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Step 1 SHAP beeswarm saved: {outimg_shap_step1}\n")

    my_params = best_params.copy()

    # ------------------ Step 1c. Importance (single model on training) ------------------ #
    print("Step 1c: Computing importance (gain / split / SHAP) on training set")

    booster = final_model.booster_
    feat_names = booster.feature_name()

    totalgain_imp = booster.feature_importance(importance_type="gain")
    totalgain_imp = dict(zip(feat_names, totalgain_imp.tolist()))
    totalgain_imp = normal_imp(totalgain_imp)

    totalcover_imp = booster.feature_importance(importance_type="split")
    totalcover_imp = dict(zip(feat_names, totalcover_imp.tolist()))
    totalcover_imp = normal_imp(totalcover_imp)

    explainer_imp = shap.TreeExplainer(final_model)
    shap_values_all = explainer_imp.shap_values(X_train)
    shap_values_used = shap_values_all[0] if isinstance(shap_values_all, list) else shap_values_all
    shap_values_abs_mean = np.abs(np.mean(shap_values_used, axis=0))
    shap_values_abs_mean = shap_values_abs_mean / shap_values_abs_mean.sum()

    shap_imp_df = pd.DataFrame({"Protein": feat_names, "ShapValues_cv": shap_values_abs_mean})
    shap_imp_df.sort_values(by="ShapValues_cv", ascending=False, inplace=True)

    tg_imp_df = pd.DataFrame({"Protein": list(totalgain_imp.keys()), "TotalGain_cv": list(totalgain_imp.values())})
    tc_imp_df = pd.DataFrame({"Protein": list(totalcover_imp.keys()), "TotalCover_cv": list(totalcover_imp.values())})

    my_imp_df = pd.merge(shap_imp_df, tg_imp_df, how="left", on="Protein")
    my_imp_df = pd.merge(my_imp_df, tc_imp_df, how="left", on="Protein")
    my_imp_df["Ensemble_cv"] = (
        my_imp_df["ShapValues_cv"] + my_imp_df["TotalGain_cv"] + my_imp_df["TotalCover_cv"]
    ) / 3
    my_imp_df.sort_values(by="TotalGain_cv", ascending=False, inplace=True)
    my_imp_df = pd.merge(my_imp_df, my_f_df, how="left", on="Protein")

    outfile_sa = result_dir / "sa_Importance.csv"
    my_imp_df.to_csv(outfile_sa, index=False)
    print(f"  Saved: {outfile_sa}\n")

    # ------------------ Step 2. Reduce features by top-proportion filter ------------------ #
    print("Step 2: Reduce features (simple top-proportion filter)")

    Imp_df = pd.read_csv(outfile_sa)
    top_nb = get_imp_analy(Imp_df, top_prop)
    Imp_df = Imp_df.iloc[:top_nb, :]
    my_f_lst_reduced = Imp_df.Protein.tolist()
    print(f"  # proteins kept (top {top_prop*100:.0f}% gain): {len(my_f_lst_reduced)}")

    myout_df = Imp_df[["Protein", "TotalGain_cv", "AD_estimate", "AUC", "EntrezGeneSymbol"]].copy()
    outfile_sb = result_dir / "sb_rmMultiColinearity.csv"
    myout_df.to_csv(outfile_sb, index=False)
    print(f"  Saved: {outfile_sb}\n")

    # ------------------ Step 3. Re-Importance on reduced feature set ------------------ #
    print("Step 3: Re-estimating importance on reduced feature set + saving OOF SHAP")

    mydf3 = pd.read_csv(pca_path)
    my_f_df2 = pd.read_csv(outfile_sb)
    my_f_lst2 = my_f_df2.Protein.tolist()

    mydf3["target_y"] = mydf3[target_col].copy()
    mydf3 = mydf3.loc[mydf3["target_y"].isin(case_labels + control_labels)]
    mydf3.reset_index(inplace=True, drop=True)
    mydf3["target_y"].replace(case_labels + control_labels, [1, 0], inplace=True)

    my_X3_all = mydf3[my_f_lst2]
    y3_all = mydf3["target_y"]

    train_idx = np.load(result_dir / "train_idx.npy")
    test_idx = np.load(result_dir / "test_idx.npy")

    my_X3_all = my_X3_all.loc[:, ~my_X3_all.columns.duplicated()]
    X_train3 = my_X3_all.iloc[train_idx].reset_index(drop=True)
    y_train3 = y3_all.iloc[train_idx].reset_index(drop=True)
    X_test3 = my_X3_all.iloc[test_idx].reset_index(drop=True)
    y_test3 = y3_all.iloc[test_idx].reset_index(drop=True)

    cv_splits_train = joblib.load(result_dir / "cv_splits_train.pkl")
    nb_folds_train = len(cv_splits_train)

    best_params3 = model_val(
        X_train3,
        y_train3,
        cv_splits=cv_splits_train,
        n_trials=n_optuna_trials,
        seed=SEED,
    )

    final_model3 = LGBMClassifier(
        **best_params3, objective="binary", is_unbalance=True, random_state=SEED
    )
    final_model3.fit(X_train3, y_train3)
    final_preds3 = final_model3.predict_proba(X_test3)[:, 1]
    test_auc3 = roc_auc_score(y_test3, final_preds3)
    print(f"  Hold-out test AUC (reduced features): {test_auc3:.4f}\n")

    best_params3_path = result_dir / "best_params_step3.json"
    with open(best_params3_path, "w") as f:
        json.dump(best_params3, f, indent=2)
    best_model3_path = result_dir / "best_model_step3.pkl"
    joblib.dump(final_model3, best_model3_path)
    print(f"  Saved Step3 params to {best_params3_path}")
    print(f"  Saved Step3 model to  {best_model3_path}\n")

    my_params3 = best_params3.copy()

    # ----- CV importance accumulators -----
    tg_imp_cv = Counter()
    tc_imp_cv = Counter()
    shap_imp_cv = np.zeros(len(my_f_lst2))

    # ----- NEW: OOF SHAP matrix on training set -----
    n_train3 = X_train3.shape[0]
    n_feat3 = X_train3.shape[1]
    shap_oof = np.zeros((n_train3, n_feat3), dtype=float)
    base_oof = np.zeros((n_train3,), dtype=float)

    # for later plotting with gene symbols
    feat_names_step3, protein_to_symbol_unique = make_unique_gene_symbols(list(X_train3.columns), soma_to_symbol)

    for tr_idx, val_idx in cv_splits_train:
        X_tr, X_val = X_train3.iloc[tr_idx, :], X_train3.iloc[val_idx, :]
        y_tr, y_val = y_train3.iloc[tr_idx], y_train3.iloc[val_idx]

        my_lgb = LGBMClassifier(
            objective="binary",
            metric="auc",
            is_unbalance=True,
            verbosity=1,
            seed=SEED,
        )
        my_lgb.set_params(**my_params3)
        my_lgb.fit(X_tr, y_tr)

        # gain/split importance
        totalgain_imp = my_lgb.booster_.feature_importance(importance_type="gain")
        totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
        totalcover_imp = my_lgb.booster_.feature_importance(importance_type="split")
        totalcover_imp = dict(zip(my_lgb.booster_.feature_name(), totalcover_imp.tolist()))
        tg_imp_cv += Counter(normal_imp(totalgain_imp))
        tc_imp_cv += Counter(normal_imp(totalcover_imp))

        # SHAP on validation fold
        explainer = shap.TreeExplainer(my_lgb)
        shap_vals = explainer.shap_values(X_val)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]  # class 0 for binary
        # store OOF SHAP
        shap_oof[val_idx, :] = shap_vals

        # base values
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev0 = float(ev[0])
        else:
            ev0 = float(ev)
        base_oof[val_idx] = ev0

        # CV SHAP importance (as you did before): mean abs SHAP on val fold
        shap_abs = np.abs(np.mean(shap_vals, axis=0))
        shap_imp_cv += shap_abs / np.sum(shap_abs)

    # Save Step 3 OOF SHAP to disk (train-only)
    shap_npz_path = result_dir / "sc_shap_oof_train.npz"
    np.savez_compressed(
        shap_npz_path,
        values=shap_oof,
        base_values=base_oof,
        data=X_train3.values,
        feature_names=np.array(feat_names_step3, dtype=object),
        protein_ids=np.array(list(X_train3.columns), dtype=object),
    )

    shap_meta_path = result_dir / "sc_shap_oof_train_meta.json"
    with open(shap_meta_path, "w") as f:
        json.dump(
            {
                "note": "OOF SHAP values computed on training folds only (Step 3).",
                "feature_names": feat_names_step3,
                "protein_ids": list(X_train3.columns),
                "protein_to_symbol_unique": protein_to_symbol_unique,
            },
            f,
            indent=2,
        )

    print(f"  Saved Step3 OOF SHAP: {shap_npz_path}")
    print(f"  Saved Step3 OOF SHAP meta: {shap_meta_path}\n")

    # ----- Build Step 3 importance table (as before) -----
    shap_imp_df = pd.DataFrame(
        {"Protein": my_f_lst2, "ShapValues_cv": shap_imp_cv / nb_folds_train}
    )
    shap_imp_df.sort_values(by="ShapValues_cv", ascending=False, inplace=True)

    tg_imp_cv = normal_imp(tg_imp_cv)
    tg_imp_df = pd.DataFrame({"Protein": list(tg_imp_cv.keys()), "TotalGain_cv": list(tg_imp_cv.values())})

    tc_imp_cv = normal_imp(tc_imp_cv)
    tc_imp_df = pd.DataFrame({"Protein": list(tc_imp_cv.keys()), "TotalCover_cv": list(tc_imp_cv.values())})

    my_imp_df2 = pd.merge(shap_imp_df, tg_imp_df, how="left", on="Protein")
    my_imp_df2 = pd.merge(my_imp_df2, tc_imp_df, how="left", on="Protein")
    my_imp_df2["Ensemble_cv"] = (
        my_imp_df2["ShapValues_cv"]
        + my_imp_df2["TotalGain_cv"]
        + my_imp_df2["TotalCover_cv"]
    ) / 3
    my_imp_df2.sort_values(by="TotalGain_cv", ascending=False, inplace=True)

    my_f_df2 = my_f_df2.drop("TotalGain_cv", axis=1, errors="ignore")
    my_imp_df2 = pd.merge(my_imp_df2, my_f_df2, how="left", on="Protein")

    outfile_sc = result_dir / "sc_ReImportance.csv"
    my_imp_df2.to_csv(outfile_sc, index=False)
    print(f"  Saved: {outfile_sc}\n")

    # ------------------ Step 4. Sequential Feature Selection (SFS) + plot ------------------ #
    print("Step 4: Sequential Feature Selection (SFS) + SFS plot")

    ImpMethod = "TotalGain"  # change to "ShapValues" if you want SHAP-based ranking
    imp_f_df = pd.read_csv(outfile_sc)
    imp_f_df.sort_values(by=ImpMethod + "_cv", ascending=False, inplace=True)
    imp_f_lst = imp_f_df.Protein.tolist()

    mydf4 = pd.read_csv(pca_path)
    mydf4["target_y"] = mydf4[target_col].copy()
    mydf4 = mydf4.loc[mydf4["target_y"].isin(case_labels + control_labels)]
    mydf4.reset_index(inplace=True, drop=True)
    mydf4["target_y"].replace(case_labels + control_labels, [1, 0], inplace=True)

    mydf4 = mydf4.loc[:, ~mydf4.columns.duplicated()]

    train_idx = np.load(result_dir / "train_idx.npy")
    test_idx = np.load(result_dir / "test_idx.npy")

    mydf4_train = mydf4.iloc[train_idx].reset_index(drop=True)
    mydf4_test = mydf4.iloc[test_idx].reset_index(drop=True)
    y4_train = mydf4_train["target_y"]
    y4_test = mydf4_test["target_y"]

    cv_splits_train = joblib.load(result_dir / "cv_splits_train.pkl")

    my_params4 = my_params3

    y_pred_lst_prev1 = np.zeros_like(y4_train).tolist()
    y_pred_lst_prev2 = np.zeros_like(y4_train).tolist()
    y_pred_lst_prev3 = np.zeros_like(y4_train).tolist()

    tmp_f = []
    AUC_cv_lst = []

    best_auc_sfs = -np.inf
    best_y_true_sfs = None
    best_y_pred_sfs = None
    best_k_sfs = None
    best_features_sfs = None

    uniqueList, duplicateList = [], []
    for i in imp_f_lst:
        if i not in uniqueList:
            uniqueList.append(i)
        elif i not in duplicateList:
            duplicateList.append(i)
    imp_f_lst = uniqueList

    for f in imp_f_lst:
        tmp_f.append(f)
        my_X_train_sfs = mydf4_train[tmp_f]
        AUC_cv, y_pred_lst, y_true_lst = [], [], []

        for tr_idx, val_idx in cv_splits_train:
            X_tr, X_val = my_X_train_sfs.iloc[tr_idx, :], my_X_train_sfs.iloc[val_idx, :]
            y_tr_fold, y_val_fold = y4_train.iloc[tr_idx], y4_train.iloc[val_idx]

            my_lgb = LGBMClassifier(
                objective="binary",
                metric="auc",
                is_unbalance=True,
                n_jobs=4,
                verbosity=1,
                seed=SEED,
            )
            my_lgb.set_params(**my_params4)
            my_lgb.fit(X_tr, y_tr_fold)
            y_pred_prob = my_lgb.predict_proba(X_val)[:, 1]

            AUC_cv.append(roc_auc_score(y_val_fold, y_pred_prob))
            y_pred_lst += y_pred_prob.tolist()
            y_true_lst += y_val_fold.tolist()

        auc_full = roc_auc_score(y_true_lst, y_pred_lst)

        log10_p1 = delong_roc_test(np.array(y_true_lst), np.array(y_pred_lst_prev1), np.array(y_pred_lst))
        log10_p2 = delong_roc_test(np.array(y_true_lst), np.array(y_pred_lst_prev2), np.array(y_pred_lst))
        log10_p3 = delong_roc_test(np.array(y_true_lst), np.array(y_pred_lst_prev3), np.array(y_pred_lst))

        y_pred_lst_prev3 = y_pred_lst_prev2
        y_pred_lst_prev2 = y_pred_lst_prev1
        y_pred_lst_prev1 = y_pred_lst

        tmp_out = np.array(
            [
                np.mean(AUC_cv),
                np.std(AUC_cv),
                10 ** log10_p1[0][0],
                10 ** log10_p2[0][0],
                10 ** log10_p3[0][0],
                auc_full,
            ]
        )
        AUC_cv_lst.append(tmp_out)
        print("   ", f, tmp_out)

        if auc_full > best_auc_sfs:
            best_auc_sfs = auc_full
            best_y_true_sfs = np.array(y_true_lst)
            best_y_pred_sfs = np.array(y_pred_lst)
            best_k_sfs = len(tmp_f)
            best_features_sfs = tmp_f.copy()

    AUC_df = pd.DataFrame(
        AUC_cv_lst, columns=["AUC_mean", "AUC_std", "Delong1", "Delong2", "Delong3", "AUC_all"]
    )
    AUC_df[["AUC_mean", "AUC_std", "AUC_all"]] = np.round(AUC_df[["AUC_mean", "AUC_std", "AUC_all"]], 3)
    AUC_df = pd.concat((pd.DataFrame({"Protein": tmp_f}), AUC_df), axis=1)
    AUC_df = pd.merge(AUC_df, imp_f_df, how="left", on="Protein")

    AUC_df = AUC_df[
        [
            "Protein",
            "AUC_mean",
            "AUC_std",
            "Delong1",
            "Delong2",
            "Delong3",
            "AUC_all",
            "TotalGain_cv",
            "AD_estimate",
            "AUC",
            "EntrezGeneSymbol",
        ]
    ]

    outfile_sd = result_dir / "sd_SFS.csv"
    AUC_df.to_csv(outfile_sd, index=False)
    print(f"  Saved: {outfile_sd}\n")

    pro_auc_df = pd.read_csv(outfile_sd)
    nb_sel = choose_k_by_auc_gain_plateau(
        pro_auc_df,
        delta=plateau_delta,
        patience=plateau_patience,
        auc_col="AUC_all",
    )
    selected_proteins = pro_auc_df["Protein"].tolist()[:nb_sel]
    selected_set = set(selected_proteins)
    print(f"  Selected k by AUC plateau: k={nb_sel} (delta={plateau_delta}, patience={plateau_patience})")

    with open(result_dir / "k_selection_plateau.json", "w") as f:
        json.dump(
            {
                "k_selected": int(nb_sel),
                "delta": float(plateau_delta),
                "patience": int(plateau_patience),
                "auc_col": "AUC_all",
                "selected_proteins": selected_proteins,
            },
            f,
            indent=2,
        )

    if best_y_true_sfs is not None:
        y_bin_sfs = (best_y_pred_sfs >= 0.5).astype(int)
        sfs_metrics_train = {
            "best_k_features_auc_max": int(best_k_sfs),
            "roc_auc": float(roc_auc_score(best_y_true_sfs, best_y_pred_sfs)),
            "accuracy_0.5": float(accuracy_score(best_y_true_sfs, y_bin_sfs)),
            "f1_0.5": float(f1_score(best_y_true_sfs, y_bin_sfs)),
            "precision_0.5": float(precision_score(best_y_true_sfs, y_bin_sfs)),
            "recall_0.5": float(recall_score(best_y_true_sfs, y_bin_sfs)),
            "n_samples": int(len(best_y_true_sfs)),
        }
        with open(result_dir / "metrics_step4_sfs_best_train.json", "w") as f:
            json.dump(sfs_metrics_train, f, indent=2)

        pred_df_sfs_train = pd.DataFrame({"y_true": best_y_true_sfs, "y_pred_prob": best_y_pred_sfs})
        pred_df_sfs_train.to_csv(result_dir / "predictions_step4_oof_bestk_train.csv", index=False)
        print("  Saved SFS best-k training metrics and OOF predictions.\n")

    if len(selected_proteins) > 0:
        X_train_plateau = mydf4_train[selected_proteins]
        X_test_plateau = mydf4_test[selected_proteins]

        final_plateau_model = LGBMClassifier(
            **my_params4, objective="binary", is_unbalance=True, random_state=SEED
        )
        final_plateau_model.fit(X_train_plateau, y4_train)

        y_test_panel = final_plateau_model.predict_proba(X_test_plateau)[:, 1]
        y_test_panel_bin = (y_test_panel >= 0.5).astype(int)
        auc_panel_test = roc_auc_score(y4_test, y_test_panel)

        sfs_metrics_test = {
            "plateau_k_features": int(nb_sel),
            "roc_auc": float(auc_panel_test),
            "accuracy_0.5": float(accuracy_score(y4_test, y_test_panel_bin)),
            "f1_0.5": float(f1_score(y4_test, y_test_panel_bin)),
            "precision_0.5": float(precision_score(y4_test, y_test_panel_bin)),
            "recall_0.5": float(recall_score(y4_test, y_test_panel_bin)),
            "n_test": int(len(y4_test)),
        }
        with open(result_dir / "metrics_step4_plateau_best_test.json", "w") as f:
            json.dump(sfs_metrics_test, f, indent=2)

        pred_df_sfs_test = pd.DataFrame({"idx": test_idx, "y_true": y4_test.values, "y_pred_prob": y_test_panel})
        pred_df_sfs_test.to_csv(result_dir / "predictions_step4_plateau_best_test.csv", index=False)
        print("  Saved PLATEAU best-k TEST metrics and predictions.\n")

    # ---- SFS Plot ----
    print("  Making SFS plot (importance bars + cumulative AUC curve)")

    pro_imp_df = pd.read_csv(outfile_sc, usecols=["Protein", "TotalGain_cv"])
    pro_imp_df.rename(columns={"TotalGain_cv": "Pro_imp"}, inplace=True)

    mydf_plot = pd.merge(pro_imp_df, pro_auc_df[["Protein", "EntrezGeneSymbol"]], how="left", on="Protein")
    mydf_plot = (
        mydf_plot.sort_values("Pro_imp", ascending=False)
        .drop_duplicates(subset="EntrezGeneSymbol", keep="first")
    )

    fig, ax = plt.subplots(figsize=(18, 6.5))
    palette = sns.color_palette("Blues", n_colors=len(mydf_plot))
    palette.reverse()

    sns.barplot(
        ax=ax,
        x="EntrezGeneSymbol",
        y="Pro_imp",
        palette=palette,
        errorbar=None,
        data=mydf_plot,
    )

    y_imp_up_lim = float(mydf_plot["Pro_imp"].max() + 0.01)
    ax.set_ylim([0, y_imp_up_lim])
    ax.tick_params(axis="y", labelsize=14)

    ax.set_xticklabels(
        mydf_plot["EntrezGeneSymbol"],
        rotation=45,
        fontsize=10,
        horizontalalignment="right",
    )

    symbol_to_protein = dict(zip(mydf_plot["EntrezGeneSymbol"], mydf_plot["Protein"]))
    for tick in ax.get_xticklabels():
        sym = tick.get_text()
        prot = symbol_to_protein.get(sym, None)
        tick.set_color("red" if prot in selected_set else "black")

    ax.set_ylabel("Protein importance", weight="bold", fontsize=18)
    ax.set_xlabel("")
    ax.grid(which="minor", alpha=0.2, linestyle=":")
    ax.grid(which="major", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    x_all = np.arange(1, len(pro_auc_df) + 1)
    auc_all = pro_auc_df["AUC_all"].astype(float).values
    ax2.plot(x_all, auc_all, color="black", alpha=0.8, marker="o")
    if nb_sel > 0:
        ax2.plot(x_all[:nb_sel], auc_all[:nb_sel], color="red", alpha=0.9, marker="o", linewidth=2)

    auc_lower = (pro_auc_df["AUC_mean"].astype(float) - pro_auc_df["AUC_std"].astype(float)).values
    auc_upper = (pro_auc_df["AUC_mean"].astype(float) + pro_auc_df["AUC_std"].astype(float)).values
    auc_upper = np.clip(auc_upper, None, 1.0)
    ax2.fill_between(x_all, auc_lower, auc_upper, color="tomato", alpha=0.2)

    ax2.set_ylabel("Cumulative AUC (OOF)", weight="bold", fontsize=18)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.set_ylim([float(np.min(auc_lower) - 0.01), float(np.max(auc_upper) + 0.01)])

    fig.tight_layout()
    outimg_sfs = result_dir / "sd_SFS_plot.png"
    plt.savefig(outimg_sfs, dpi=200)
    plt.close(fig)
    print(f"  SFS plot saved: {outimg_sfs}\n")

    print(">>> Pipeline finished for", exp_name)
    print()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM-based protein selection pipeline")
    parser.add_argument("--base-dir", default=".", help="Base directory")
    parser.add_argument(
        "--soma-annotation-file",
        default="0.soma_annotation.csv",
        help="SomaLogic annotation file",
    )
    parser.add_argument("--output-root", default="Final", help="Root output directory")
    parser.add_argument("--n-optuna-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--top-prop", type=float, default=0.75, help="Top proportion of gain used in Step 2 (0–1)")
    parser.add_argument("--seed", type=int, default=2025, help="Global random seed (LightGBM, Optuna, CV)")

    parser.add_argument(
        "--delong-col",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="(Legacy) Which DeLong column (1, 2, or 3). DeLong still computed but not used for k selection.",
    )
    parser.add_argument("--plateau-delta", type=float, default=0.005, help="AUC gain threshold for plateau detection.")
    parser.add_argument("--plateau-patience", type=int, default=2, help="Consecutive small-gain steps for plateau.")

    parser.add_argument(
        "--dap-files",
        nargs="+",
        default=[
            "DAPs/2.LG_AD_A+_vs_CN_A-_agesex.csv",
            "DAPs/2.LG_MCI_A+_vs_CN_A-_agesex.csv",
            "DAPs/3.LG_A+_vs_CN_A-_agesex.csv",
        ],
        help="One or more DAP CSV paths (space-separated).",
    )
    parser.add_argument(
        "--target-cols",
        nargs="+",
        default=["DGstatus2", "DGstatus2", "DGstatus4"],
        help="Target column for each DAP file (must match --dap-files length).",
    )
    parser.add_argument(
        "--pca-files",
        nargs="+",
        default=["modified_CSV/0.pca_modified_gapdf.csv"],
        help="One or more PCA CSV paths (space-separated).",
    )
    parser.add_argument(
        "--case-labels",
        nargs="+",
        default=None,
        help="Labels treated as cases (1) (optional; normally inferred from DAP filename).",
    )
    parser.add_argument(
        "--control-labels",
        nargs="+",
        default=None,
        help="Labels treated as controls (0) (optional; normally inferred from DAP filename).",
    )

    args = parser.parse_args()

    SEED = args.seed
    set_global_seed(SEED)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    pca_files = args.pca_files
    dap_files = args.dap_files
    target_cols = args.target_cols

    if len(dap_files) != len(target_cols):
        raise ValueError(
            f"--dap-files (n={len(dap_files)}) and --target-cols (n={len(target_cols)}) must have the same length."
        )

    for pca_file in pca_files:
        for dap_file, target_col in zip(dap_files, target_cols):
            if args.case_labels is not None and args.control_labels is not None:
                case_labels = tuple(args.case_labels)
                control_labels = tuple(args.control_labels)
            else:
                case_labels, control_labels = infer_labels_from_dap_path(dap_file)

            print(f"Running now: {pca_file} | {dap_file}")
            print(f"  case_labels={case_labels}, control_labels={control_labels}")
            print(f"  target_col={target_col}")

            run_protein_selection_experiment(
                base_dir=args.base_dir,
                pca_file=pca_file,
                soma_annotation_file=args.soma_annotation_file,
                dap_file=dap_file,
                target_col=target_col,
                case_labels=case_labels,
                control_labels=control_labels,
                output_root=args.output_root,
                n_optuna_trials=args.n_optuna_trials,
                top_prop=args.top_prop,
                delong_col=args.delong_col,
                ts=ts,
                plateau_delta=args.plateau_delta,
                plateau_patience=args.plateau_patience,
            )
