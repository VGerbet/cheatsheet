from src.xgboost.classifier import XGBClassifier, XGBClassifierMulti
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder


def test_classifier_optimizer(df_classification):
    study = XGBClassifier(
        df_classification.drop("survived", axis=1), df_classification["survived"]
    ).optimize(2)
    dc = DummyClassifier().fit(
        df_classification.drop("survived", axis=1), df_classification["survived"]
    )
    dc_score = roc_auc_score(
        df_classification["survived"],
        dc.predict(df_classification.drop("survived", axis=1)),
    )
    print(f"DummyClassifier score: {dc_score}, XGBClassifier score: {study.best_value}")
    assert True


def test_classifiermulti_optimizer(df_classification_multi):
    study = XGBClassifierMulti(
        df_classification_multi.drop("species", axis=1),
        LabelEncoder().fit_transform(df_classification_multi["species"]),
    ).optimize(2)
    dc = DummyClassifier(strategy="stratified").fit(
        df_classification_multi.drop("species", axis=1),
        LabelEncoder().fit_transform(df_classification_multi["species"]),
    )
    dc_score = log_loss(
        LabelEncoder().fit_transform(df_classification_multi["species"]),
        dc.predict_proba(df_classification_multi.drop("species", axis=1)),
    )
    print(f"DummyClassifier score: {dc_score}, XGBClassifier score: {study.best_value}")
    assert True
