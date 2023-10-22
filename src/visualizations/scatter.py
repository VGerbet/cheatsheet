import pandas as pd
import plotly.express as px
import numpy as np


def plot_scatters(df: pd.DataFrame):
    fig = px.scatter_matrix(df.select_dtypes(include=np.number))
    fig.show()


def plot_interaction(df: pd.DataFrame, col1: str, col2: str, target: str):
    fig = px.scatter(
        x=df[col1],
        y=df[col2],
        color=df[target],
        labels={"x": col1, "y": col2, "color": target},
        # range_x=(0, 10),
    )
    fig.show()
