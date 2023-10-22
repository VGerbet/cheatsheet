import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlations(df: pd.DataFrame):
    corr = (
        df.select_dtypes(include=np.number)
        .corr()
        .fillna(0)
        .reset_index()
        .melt(id_vars="index")
    )
    fig = px.imshow(
        corr.pivot(index="index", columns="variable", values="value"),
        color_continuous_scale="Reds",
    )
    fig.update_xaxes(side="top")
    fig.show()


def plot_correlations_dendrogram(
    df: pd.DataFrame, threshold: float = 0.5, width: int = 20, height: int = 20
):
    corr = df.select_dtypes(include=np.number).corr().fillna(0)
    col_to_keep = corr[(corr.abs() > threshold).sum() > 1].index
    corr = corr.loc[col_to_keep, col_to_keep]
    mask = corr < threshold
    cg = sns.clustermap(
        corr,
        annot=True,
        fmt=".2f",
        method="single",
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        vmin=-1,
        vmax=1,
        figsize=(width, height),
        mask=mask,
        linewidths=0.5,
        linecolor="black",
    )
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    cg.cax.set_visible(False)
    plt.show()
