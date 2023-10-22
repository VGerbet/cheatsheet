import pandas as pd
import seaborn as sns
import pytest


# Load a dataframe as a fixture for regression tests
@pytest.fixture(scope="module")
def df_regression():
    return sns.load_dataset("diamonds")


@pytest.fixture(scope="module")
def df_classification():
    df = sns.load_dataset("titanic")
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].astype("category")
    return df


@pytest.fixture(scope="module")
def df_classification_multi():
    df = sns.load_dataset("iris")
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].astype("category")
    return df
