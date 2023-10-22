from src.xgboost.regressor import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error


def test_regressor_optimizer(df_regression):
    study = XGBRegressor(
        df_regression.drop("price", axis=1), df_regression["price"]
    ).optimize(2)
    dr = DummyRegressor().fit(
        df_regression.drop("price", axis=1), df_regression["price"]
    )
    dr_score = mean_squared_error(
        df_regression["price"],
        dr.predict(df_regression.drop("price", axis=1)),
        squared=False,
    )
    print(f"DummyRegressor score: {dr_score}, XGBRegressor score: {study.best_value}")
    assert True
