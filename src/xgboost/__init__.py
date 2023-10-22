import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error  # or any other metric
from sklearn.model_selection import train_test_split
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, cross_val_score


class XGBOptimizer(ABC):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        n_folds: int = 2,
    ) -> None:
        self.X = X
        self.y = y
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )
        self.n_folds = n_folds

    @property
    @abstractmethod
    def direction(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def objective(
        self,
        trial: optuna.trial.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        cv: KFold,
        scoring: str,
    ) -> float:
        raise NotImplementedError

    def optimize(self, n_trials: int = 100) -> optuna.study.Study:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(pruner=pruner, direction=self.direction)
        func = lambda trial: self.objective(
            trial,
            self.X_train,
            self.y_train,
        )
        study.optimize(func, n_trials=n_trials)

        self.best_params = study.best_params
        self.best_score = study.best_value
        print("Best Hyperparameters: ", self.best_params)
        print("Best score: ", self.best_score)

        return study
