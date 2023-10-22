import pandas as pd
import plotly.express as px
import numpy as np


def plot_boxplot(df: pd.DataFrame):
    fig = px.box(
        df.select_dtypes(include=np.number).melt(), y="value", facet_col="variable"
    ).update_yaxes(matches=None)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.show()
