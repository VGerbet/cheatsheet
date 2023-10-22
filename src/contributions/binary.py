import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from shap import TreeExplainer


def shap_sum2proba(shap_sum: float):
    """Compute sigmoid function of the Shap sum to get predicted probability."""

    return 1 / (1 + np.exp(-shap_sum))


def individual_log_loss(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-15):
    """Compute log-loss for each individual of the sample."""

    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def get_preds_shaps(model: XGBClassifier, X: pd.DataFrame):
    """Get predictions (predicted probabilities) and SHAP values for a dataset."""
    preds = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    shap_explainer = TreeExplainer(model)
    shap_expected_value = shap_explainer.expected_value[-1]
    shaps = pd.DataFrame(
        data=shap_explainer.shap_values(X)[1],
        index=X.index,
        columns=X.columns,
    )
    return preds, shaps, shap_expected_value


def get_feature_contributions(
    y_true: pd.Series,
    y_pred: pd.Series,
    shap_values: pd.DataFrame,
    shap_expected_value: np.ndarray,
):
    """Compute prediction contribution and error contribution for each feature."""

    prediction_contribution = shap_values.abs().mean().rename("prediction_contribution")

    ind_log_loss = individual_log_loss(y_true=y_true, y_pred=y_pred).rename("log_loss")
    y_pred_wo_feature = shap_values.apply(
        lambda feature: shap_expected_value + shap_values.sum(axis=1) - feature
    ).applymap(shap_sum2proba)
    ind_log_loss_wo_feature = y_pred_wo_feature.apply(
        lambda feature: individual_log_loss(y_true=y_true, y_pred=feature)
    )
    ind_log_loss_diff = ind_log_loss_wo_feature.apply(
        lambda feature: ind_log_loss - feature
    )
    error_contribution = ind_log_loss_diff.mean().rename("error_contribution").T

    return prediction_contribution, error_contribution
