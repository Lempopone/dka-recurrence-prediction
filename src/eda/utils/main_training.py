import numpy as np
import pandas as pd
import os
import joblib
import optuna
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
from itertools import combinations
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)




def get_models(random_state=42):
    return {
        "xgb_tree": XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            verbosity=0
        ),
        "xgb_linear": XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            verbosity=0
        ),
        "logreg": LogisticRegression(
            random_state=random_state,
            n_jobs=-1
        ),
        "lgm": LGBMClassifier(
            random_state=random_state,
            verbose=-1
        ),
        "rf": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        ),
        "dt": DecisionTreeClassifier(
            random_state=random_state
        ),
        "catboost": CatBoostClassifier(
            random_state=random_state,
            verbose=0
        ),
    }

def get_preprocessor(model_name, cat_cols, num_cols):

    if model_name == "catboost":
        return "passthrough"

    if model_name == "logreg":
        return ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ])

    elif model_name in ["rf", "dt", "lgm", "xgb_tree", "xgb_linear"]:
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ], remainder="passthrough")

    else:
        return "passthrough"

def get_optuna_params(trial, model_name):

    ### определение коэффициента баланса таргета
    scale_pos_weight = 68 / 32

    ###  XGBoost (tree)
    if model_name == "xgb_tree":
        return {
            "booster": "gbtree",
            "n_estimators": trial.suggest_int("n_estimators", 250, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 10),
            "gamma": trial.suggest_float("gamma", 0.5, 3.0),
            "subsample": trial.suggest_float("subsample", 0.6, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.85),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 8.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 8.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }

    ### XGBoost (linear)
    elif model_name == "xgb_linear":
        return {
            "booster": "gblinear",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 8.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 3.0, 8.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }

    ### Logistic Regression
    elif model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 0.01, 2.0, log=True),
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 5000,
        }

    ### LightGBM
    elif model_name == "lgm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "num_leaves": trial.suggest_int("num_leaves", 16, 48),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.25, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.25, 5.0, log=True),
            "class_weight": "balanced",
            "verbosity": -1,
        }

    ### Random Forest
    elif model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 15),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
        }

    ### Decision Tree
    elif model_name == "dt":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 25),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": "balanced",
        }

    ### CatBoost
    elif model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 900),
            "depth": trial.suggest_int("depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 7, 20),
            "random_strength": trial.suggest_float("random_strength", 1.0, 3.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 3.0),
            "rsm": trial.suggest_float("rsm", 0.7, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 150),
            "scale_pos_weight": scale_pos_weight,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
        }

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def find_best_threshold(y_true, y_pred, min_recall=0.7):
    thresholds = np.linspace(0.01, 0.99, 200)

    best_thr = 0.5
    best_precision = -1

    fallback_thr = 0.5
    best_recall_diff = float("inf")

    for thr in thresholds:
        preds = (y_pred > thr).astype(int)

        recall = recall_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)

        ### задание условие на верхний recall
        if recall >= min_recall:
            if precision > best_precision:
                best_precision = precision
                best_thr = thr

        ### ближайший recall к цели
        recall_diff = abs(recall - min_recall)
        if recall_diff < best_recall_diff:
            best_recall_diff = recall_diff
            fallback_thr = thr

    ### если условие не достигнуто
    if best_precision == -1:
        return fallback_thr, None

    return best_thr, best_precision


def train_full_pipeline_nested_cv(
        df,
        target_col,
        get_optuna_params,
        cat_cols,
        num_cols,
        save_path=r"C:\Users\olegs\PycharmProjects\dka_recurrence_project\src\eda\models",
        outer_splits=5,
        inner_splits=3,
        n_trials=30,
        min_recall=0.7
):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = cat_cols.tolist()
    num_cols = num_cols.tolist()
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = get_models()
    results = {}

    for model_name, base_model in tqdm(models.items(), desc="Models"):

        model_dir = os.path.join(save_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)

        oof = np.zeros(len(X_train_full))
        test_folds = []
        outer_metrics = []
        best_params_list = []
        thresholds = []

        for outer_fold, (outer_train_idx, outer_val_idx) in tqdm(
                enumerate(outer_cv.split(X_train_full, y_train_full)),
                total=outer_splits,
                desc=f"{model_name} outer folds"
        ):

            X_outer_train = X_train_full.iloc[outer_train_idx]
            y_outer_train = y_train_full.iloc[outer_train_idx]
            X_outer_val = X_train_full.iloc[outer_val_idx]
            y_outer_val = y_train_full.iloc[outer_val_idx]

            ### обучение на внутренней кросс-валидации
            def objective(trial):
                params = get_optuna_params(trial, model_name)

                inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
                aucs = []

                for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):

                    X_inner_train = X_outer_train.iloc[inner_train_idx]
                    y_inner_train = y_outer_train.iloc[inner_train_idx]
                    X_inner_val = X_outer_train.iloc[inner_val_idx]
                    y_inner_val = y_outer_train.iloc[inner_val_idx]

                    if model_name == "catboost":
                        model = CatBoostClassifier(**params)
                        model.fit(X_inner_train, y_inner_train, cat_features=cat_indices, verbose=0)
                        preds = model.predict_proba(X_inner_val)[:, 1]
                    else:
                        model = base_model.__class__(**params)
                        pipe = Pipeline([
                            ("prep", get_preprocessor(model_name, cat_cols, num_cols)),
                            ("model", model)
                        ])
                        pipe.fit(X_inner_train, y_inner_train)
                        preds = pipe.predict_proba(X_inner_val)[:, 1]

                    aucs.append(roc_auc_score(y_inner_val, preds))

                return np.mean(aucs)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_params_list.append(best_params)

            ### обучаем на внешней кросс-валидации
            if model_name == "catboost":
                model = CatBoostClassifier(**best_params)
                model.fit(X_outer_train, y_outer_train, cat_features=cat_indices, verbose=0)

                train_pred = model.predict_proba(X_outer_train)[:, 1]
                val_pred = model.predict_proba(X_outer_val)[:, 1]
                test_pred = model.predict_proba(X_test)[:, 1]

                model_to_save = model

            else:
                model = base_model.__class__(**best_params)
                pipe = Pipeline([
                    ("prep", get_preprocessor(model_name, cat_cols, num_cols)),
                    ("model", model)
                ])
                pipe.fit(X_outer_train, y_outer_train)

                train_pred = pipe.predict_proba(X_outer_train)[:, 1]
                val_pred = pipe.predict_proba(X_outer_val)[:, 1]
                test_pred = pipe.predict_proba(X_test)[:, 1]

                model_to_save = pipe

            ### определение оптимального трешхолда для мед задачи
            best_thr, _ = find_best_threshold(
                y_outer_train,
                train_pred,
                min_recall=min_recall
            )
            thresholds.append(best_thr)

            ### сохранение моделей по фолдам для каждой модели
            joblib.dump(
                model_to_save,
                os.path.join(model_dir, f"{model_name}_outer_fold_{outer_fold}.pkl")
            )

            ### метрики
            auc_train = roc_auc_score(y_outer_train, train_pred)
            auc_val = roc_auc_score(y_outer_val, val_pred)
            auc_test = roc_auc_score(y_test, test_pred)

            tn, fp, fn, tp = confusion_matrix(
                y_outer_val,
                (val_pred > best_thr).astype(int)
            ).ravel()

            oof[outer_val_idx] = val_pred
            test_folds.append(test_pred)

            outer_metrics.append({
                "auc_train": auc_train,
                "auc_val": auc_val,
                "auc_test": auc_test,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "threshold": best_thr
            })

        results[model_name] = {
            "oof": oof,
            "test_pred": np.mean(test_folds, axis=0),
            "metrics": outer_metrics,
            "best_params": best_params_list,
            "thresholds": thresholds
        }

    return results, X_train_full, X_test, y_train_full, y_test


def summarize_results(results, y_train, y_test):

    summary = {}

    for model_name, data in results.items():

        metrics = data["metrics"]

        auc_train = [m["auc_train"] for m in metrics]
        auc_val = [m["auc_val"] for m in metrics]
        auc_test = [m["auc_test"] for m in metrics]

        oof = data["oof"]
        test_pred = data["test_pred"]

        # медиана
        thr = np.median(data["thresholds"])

        print(f"\n===== {model_name} =====")

        print("TRAIN AUC:", np.mean(auc_train), "+/-", np.std(auc_train))
        print("VAL AUC:", np.mean(auc_val), "+/-", np.std(auc_val))
        print("TEST AUC:", np.mean(auc_test), "+/-", np.std(auc_test))

        print(f"\nMedian threshold: {thr:.4f}")

        print("\nOOF REPORT (train CV):")
        print(classification_report(y_train, (oof > thr).astype(int)))

        print("\nTEST REPORT:")
        print(classification_report(y_test, (test_pred > thr).astype(int)))

        summary[model_name] = {
            "train_auc_mean": np.mean(auc_train),
            "val_auc_mean": np.mean(auc_val),
            "test_auc_mean": np.mean(auc_test),
            "threshold_median": thr
        }

    return summary


def train_stacking_meta_model(results, model_names, y_train, y_test):


    ### стекинг признаки на трейне
    X_meta_train = np.column_stack([
        results[m]["oof"] for m in model_names
    ])

    ### стекинг признаки на тесте
    X_meta_test = np.column_stack([
        results[m]["test_pred"] for m in model_names
    ])

    meta_model = LogisticRegression(C=0.1)
    meta_model.fit(X_meta_train, y_train)

    train_pred = meta_model.predict_proba(X_meta_train)[:, 1]
    test_pred = meta_model.predict_proba(X_meta_test)[:, 1]

    print("\n=== STACKING RESULTS ===")
    print("Train AUC:", roc_auc_score(y_train, train_pred))
    print("Test AUC:", roc_auc_score(y_test, test_pred))

    return meta_model


def search_best_stacking_combinations(results, y_train, y_test):

    model_names = ["catboost", "dt", "rf", "lgm", "logreg", "xgb_linear", "xgb_tree"]

    all_results = []

    # перебор размеров ансамбля
    for k in range(2, len(model_names) + 1):

        for combo in combinations(model_names, k):

            ### сбор метапризнаков из предсказаний
            X_meta_train = np.column_stack([
                results[m]["oof"] for m in combo
            ])

            X_meta_test = np.column_stack([
                results[m]["test_pred"] for m in combo
            ])

            ### собираем мета-модель
            meta_model = LogisticRegression(max_iter=1000)
            meta_model.fit(X_meta_train, y_train)

            test_pred = meta_model.predict_proba(X_meta_test)[:, 1]
            auc = roc_auc_score(y_test, test_pred)

            all_results.append({
                "models": combo,
                "n_models": k,
                "test_auc": auc
            })

    ### сортируем по лучшим стекингам
    all_results = sorted(all_results, key=lambda x: x["test_auc"], reverse=True)

    return all_results


def run_catboost_shap_ensemble(model_dir, X_test):

    print("\n CATBOOST SHAP (ENSEMBLE OF 5 FOLDS)")

    shap_values_all = []

    # загружаем все 5 моделей
    for fold in range(5):
        model_path = os.path.join(
            model_dir,
            "catboost",
            f"catboost_outer_fold_{fold}.pkl"
        )

        model = joblib.load(model_path)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_values_all.append(shap_values)

    # усредняем SHAP значения
    shap_values_mean = np.mean(shap_values_all, axis=0)

    ### графики шап- анализа
    print("Summary (dot)")
    shap.summary_plot(
        shap_values_mean,
        X_test,
        plot_type="dot",
        max_display=15
    )

    print("Feature importance (bar)")
    shap.summary_plot(
        shap_values_mean,
        X_test,
        plot_type="bar",
        max_display=15
    )