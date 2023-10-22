import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error  # or any other metric
from sklearn.model_selection import train_test_split
from src.xgboost import XGBOptimizer
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score


class XGBRegressor(XGBOptimizer):
    @property
    def direction(self) -> str:
        return "minimize"

    # Define the objective function for Optuna
    def objective(
        self,
        trial: optuna.trial.Trial,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        # Define the search space for hyperparameters
        param = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": trial.suggest_float("eta", 0.01, 0.3),
            "num_boost_round": 100000,  # Fix the boosting round and use early stopping
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
            "lambda": trial.suggest_float("lambda", 0.1, 10.0),
            "alpha": trial.suggest_float("alpha", 0.0, 10.0),
        }

        # Convert the data into DMatrix format
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

        # Define the pruning callback for early stopping
        # Seems broken for now:
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")

        history = xgb.cv(
            param,
            dtrain,
            num_boost_round=100,
            nfold=self.n_folds,
            early_stopping_rounds=50,
        )  # callbacks=[pruning_callback])

        trial.set_user_attr("n_estimators", len(history))

        # Calculate the root mean squared error
        rmse = history["test-rmse-mean"].iloc[-1]

        return rmse
