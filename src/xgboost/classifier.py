import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error  # or any other metric
from sklearn.model_selection import train_test_split
from src.xgboost import XGBOptimizer
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score


class XGBClassifier(XGBOptimizer):
    @property
    def direction(self) -> str:
        return "maximize"

    # Define the objective function for Optuna
    def objective(
        self,
        trial: optuna.trial.Trial,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        # Define the search space for hyperparameters
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        # Convert the data into DMatrix format
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

        # Define the pruning callback for early stopping
        # Seems broken for now:
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")

        history = xgb.cv(
            param,
            dtrain,
            num_boost_round=100,
            stratified=True,
        )  # callbacks=[pruning_callback])

        trial.set_user_attr("n_estimators", len(history))

        best_score = history["test-auc-mean"].values[-1]

        return best_score


class XGBClassifierMulti(XGBOptimizer):
    @property
    def direction(self) -> str:
        return "maximize"

    # Define the objective function for Optuna
    def objective(
        self,
        trial: optuna.trial.Trial,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        # Define the search space for hyperparameters
        param = {
            "verbosity": 0,
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "num_class": y.max() + 1,
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        # Convert the data into DMatrix format
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

        # Define the pruning callback for early stopping
        # Seems broken for now:
        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, "test-mlogloss"
        )

        history = xgb.cv(
            param,
            dtrain,
            num_boost_round=100,
            stratified=True,
        )  # callbacks=[pruning_callback])

        trial.set_user_attr("n_estimators", len(history))

        best_score = history["test-mlogloss-mean"].values[-1]

        return best_score
