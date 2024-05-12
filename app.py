import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Added this line
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display
import numpy as np  # Added this line as np is also used in the code

vehicles = pd.read_csv('vehicles_us.csv')

print(vehicles.head())
print(vehicles.info())

fig_histogram = px.histogram(vehicles, x='price', title='Vehicle Price Distribution')
fig_histogram.show()

fig_scatter = px.scatter(vehicles, x='model_year', y='price', title='Price vs. Model Year')

checkbox = widgets.Checkbox(
    value=True,
    description='Show Trend Line',
    disabled=False
)

fig_scatter = go.FigureWidget(data=[
    go.Scatter(x=vehicles['model_year'], y=vehicles['price'], mode='markers', name='Price vs. Model Year',
               marker=dict(color='blue', size=8, opacity=0.5))
])


def update_scatter_plot(change):
    with fig_scatter.batch_update():
        fig_scatter.data = []
        fig_scatter.add_trace(
            go.Scatter(x=vehicles['model_year'], y=vehicles['price'], mode='markers', name='Price vs. Model Year',
                       marker=dict(color='blue', size=8, opacity=0.5)))
        if change['new']:
            valid_data = vehicles[['model_year', 'price']].dropna()
            if len(valid_data) > 1:
                z = np.polyfit(valid_data['model_year'], valid_data['price'], 1)
                p = np.poly1d(z)
                sorted_model_years = np.sort(valid_data['model_year'].unique())
                fig_scatter.add_trace(
                    go.Scatter(x=sorted_model_years, y=p(sorted_model_years), mode='lines', name='Trend Line',
                               line=dict(color='red', width=2)))


fig_scatter.update_layout(title='Price vs. Model Year', xaxis_title='Model Year', yaxis_title='Price',
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))

checkbox.observe(update_scatter_plot, names='value')
display(checkbox)
update_scatter_plot({'new': checkbox.value})
fig_scatter.show()

plotobservations1 = """Observations from the scatter plot indicate a slight upward trend, suggesting newer cars tend to be priced higher, which aligns with expectations. Several outliers, particularly in newer model years with exceptionally high prices, could be luxury or special edition models. The price range widens as the model year increases, suggesting the availability of more premium models in newer years. Additionally, a higher density of data points in the most recent model years indicates a higher volume of transactions."""

print(plotobservations1)
