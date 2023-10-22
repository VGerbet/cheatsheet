import pandas as pd
import plotly.express as px


def plot_pie(s: pd.Series):
    # plot pie for embarked_town using plotly
    fig = px.pie(
        s,
        values=s.value_counts(dropna=False).values,
        names=s.value_counts(dropna=False).index,
        title=s.name,
    )
    fig.show()
