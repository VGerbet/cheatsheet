from xgboost import XGBRegressor
import pandas as pd
from shap import TreeExplainer


def get_preds_shaps(model: XGBRegressor, X: pd.DataFrame):
    """
    Args:
        model: Trained XGBRegressor
        X: pd.DataFrame set to predict
    """
    preds = pd.Series(model.predict(X, index=X.index))
    shap_explainer = TreeExplainer(model)
    shaps = pd.DataFrame(
        data=shap_explainer.shap_values(X), index=X.index, columns=X.columns
    )
    return preds, shaps


def get_feature_contributions(
    y_true: pd.Series, y_pred: pd.Series, shap_values: pd.DataFrame
):
    """Compute prediction contribution and error contribution for each feature."""

    prediction_contribution = shap_values.abs().mean().rename("prediction_contribution")

    abs_error = (y_true - y_pred).abs()
    y_pred_wo_feature = shap_values.apply(lambda feature: y_pred - feature)
    abs_error_wo_feature = y_pred_wo_feature.apply(
        lambda feature: (y_true - feature).abs()
    )
    error_contribution = (
        abs_error_wo_feature.apply(lambda feature: abs_error - feature)
        .mean()
        .rename("error_contribution")
    )

    return prediction_contribution, error_contribution
