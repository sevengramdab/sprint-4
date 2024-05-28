import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display
import numpy as np
import seaborn as sns
import os
from ipywidgets import Checkbox
from ipywidgets import interactive



print(os.getcwd())


# ... the rest of your code...

#streamlit run C:/Program Files/JetBrains/PyCharm 2024.1/plugins/python/helpers/pydev/pydevconsole.py

sns.set(style="whitegrid")
vehicles = pd.read_csv('vehicles_us.csv')

print(vehicles.head())
print(vehicles.info())

import pandas as pd

# Load the dataset
vehicles_df = pd.read_csv('vehicles_us.csv')
# Display the first few rows of the dataframe
display(vehicles_df.head())
# %%
from ipywidgets import interactive, HBox, VBox, widgets
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go

df = pd.read_csv('vehicles_us.csv')
df['manufacturer'] = df['model'].apply(lambda x: x.split()[0])

st.header('Data viewer')
show_manuf_1k_ads = st.checkbox('Include manufacturers with less than 1000 ads')
if not show_manuf_1k_ads:
    df = df.groupby('manufacturer').filter(lambda x: len(x) > 1000)

st.dataframe(df)
st.header('Vehicle types by manufacturer')
st.write(px.histogram(df, x='manufacturer', color='type'))
st.header('Histogram of `condition` vs `model_year`')

# -------------------------------------------------------
# histograms in plotly:
# fig = go.Figure()
# fig.add_trace(go.Histogram(x=df[df['condition']=='good']['model_year'], name='good'))
# fig.add_trace(go.Histogram(x=df[df['condition']=='excellent']['model_year'], name='excellent'))
# fig.update_layout(barmode='stack')
# st.write(fig)
# works, but too many lines of code
# -------------------------------------------------------

# histograms in plotly_express:
st.write(px.histogram(df, x='model_year', color='condition'))
# a lot more concise!
# -------------------------------------------------------

st.header('Compare price distribution between manufacturers')
manufac_list = sorted(df['manufacturer'].unique())
manufacturer_1 = st.selectbox('Select manufacturer 1',
                              manufac_list, index=manufac_list.index('chevrolet'))

manufacturer_2 = st.selectbox('Select manufacturer 2',
                              manufac_list, index=manufac_list.index('hyundai'))
mask_filter = (df['manufacturer'] == manufacturer_1) | (df['manufacturer'] == manufacturer_2)
df_filtered = df[mask_filter]
normalize = st.checkbox('Normalize histogram', value=True)
if normalize:
    histnorm = 'percent'
else:
    histnorm = None
st.write(px.histogram(df_filtered,
                      x='price',
                      nbins=30,
                      color='manufacturer',
                      histnorm=histnorm,
                      barmode='overlay'))

# Function to update the histogram based on the trend line visibility
def update_histogram(show_trendline):
    fig = px.histogram(vehicles_df, x='price', title='Vehicle Price Distribution')
    if show_trendline:
        fig.add_traces(go.Scatter(x=np.sort(vehicles_df['price']),
                                  y=np.poly1d(np.polyfit(vehicles_df['price'],
                                                         np.histogram(vehicles_df['price'], bins=40)[0],
                                                         1))(np.sort(vehicles_df['price'])),
                                  mode='lines', name='Trend Line'))
    fig.show()


# Ensure there are no duplicate options by using unique widget identifiers if needed

# Create interactive widget
interactive_plot = interactive(update_histogram,
                               show_trendline=widgets.Checkbox(value=False, description='Show Trend Line'))
display(interactive_plot)
# %%
from ipywidgets import interactive, HBox, VBox, widgets, Layout
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# Function to update the histogram based on the trend line, outlier removal options, and ensuring no negative y values
def update_histogram(show_trendline, remove_outliers):
    # Filter out outliers if the checkbox is checked
    if remove_outliers:
        Q1 = vehicles_df['price'].quantile(0.25)
        Q3 = vehicles_df['price'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_df = vehicles_df.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 * @IQR)')
    else:
        filtered_df = vehicles_df

    fig = px.histogram(filtered_df, x='price', title='Vehicle Price Distribution')

    if show_trendline:
        # Calculate and display the trend line
        # Ensure no negative y values by clipping at 0
        y_values = np.poly1d(np.polyfit(filtered_df['price'],
                                        np.histogram(filtered_df['price'], bins=40)[0],
                                        1))(np.sort(filtered_df['price']))
        y_values_clipped = np.clip(y_values, a_min=0, a_max=None)  # Clip negative values to 0
        fig.add_traces(go.Scatter(x=np.sort(filtered_df['price']),
                                  y=y_values_clipped,
                                  mode='lines', name='Trend Line'))
    fig.update_layout(showlegend=True)
    fig.show()


# Create interactive widgets with specified layout for better appearance
checkbox_trendline = widgets.Checkbox(value=False, description='Show Trend Line',
                                      layout=Layout(width='auto', margin='0 0 0 20px'))
checkbox_outliers = widgets.Checkbox(value=False, description='Remove Outliers',
                                     layout=Layout(width='auto', margin='0 0 0 20px'))

# Layout adjustments for better UI
ui = VBox([checkbox_trendline, checkbox_outliers])

interactive_plot = interactive(update_histogram,
                               show_trendline=checkbox_trendline,
                               remove_outliers=checkbox_outliers)

# Display the UI and interactive plot together
display(ui, interactive_plot)
# %%
import pandas as pd
import plotly.express as px

# Assuming vehicles_df_filtered is already defined in the environment
# Generate a bar plot for Average Vehicle Price vs. Model Year by Type with an interactive legend
# Limiting the plot to vehicles below 200k miles, limiting y axis to below 100,000, and only showing years from 1960 onward

# First, filter the dataframe for vehicles below 200k miles
vehicles_below_200k = vehicles_df[vehicles_df['odometer'] < 200000]

# Then, calculate the average price per model year and type
avg_price_by_year_type = vehicles_below_200k.groupby(['model_year', 'type'])['price'].mean().reset_index()

fig = px.bar(avg_price_by_year_type, x='model_year', y='price', color='type',
             labels={'price': 'Average Vehicle Price', 'model_year': 'Model Year'},
             title='Average Vehicle Price vs. Model Year by Type (Below 300k Miles)')

# Update layout to improve visibility and interactivity, including limiting y axis to below 100,000 and only showing years from 1960 onward
fig.update_layout(legend_title_text='Vehicle Type',
                  xaxis=dict(
                      title='Model Year',
                      range=[1960, max(avg_price_by_year_type['model_year']) + 1]  # Only show years from 1960 onward
                  ),
                  yaxis=dict(
                      title='Average Vehicle Price',
                      range=[0, 300000]  # Limiting y axis to below 100,000
                  ),
                  legend=dict(
                      title_font_size=12,
                      itemclick="toggleothers",
                      itemdoubleclick="toggle",
                      # Enable selecting multiple legend items by setting 'groupclick' to 'togglegroup'
                      groupclick="togglegroup"
                  ),
                  font=dict(
                      family="Arial, sans-serif",
                      size=12,
                      color="RebeccaPurple"
                  ))

fig.show()

# %%
from ipywidgets import interactive, HBox, VBox, widgets
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# Function to update the histogram based on the trend line visibility, x-axis limit, bin size, and additional plotted lines
def update_histogram(show_trendline, bins, show_median_line, show_mean_line):
    fig = px.histogram(vehicles_df, x='price', title='Vehicle Price Distribution', nbins=bins)
    fig.update_xaxes(range=[0, 50000])  # Limiting x-axis to 50k
    # Update to make each bar alternating colors
    fig.update_traces(marker=dict(color=['indianred', 'lightblue'] * int(bins / 2)), marker_line_width=1.5,
                      opacity=0.75)
    if show_trendline:
        # Filter data to include only prices up to 50k for trend line calculation
        filtered_data = vehicles_df[vehicles_df['price'] <= 50000]
        # Calculate histogram data manually to avoid issues with trend line calculation
        hist_data, bin_edges = np.histogram(filtered_data['price'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Fit a polynomial of degree 1 (linear fit) to the histogram data
        coeffs = np.polyfit(bin_centers, hist_data, 1)
        poly = np.poly1d(coeffs)
        # Generate x values from the minimum to the maximum bin center for plotting the trend line
        x_trend = np.linspace(min(bin_centers), max(bin_centers), 500)
        y_trend = poly(x_trend)
        fig.add_traces(go.Scatter(x=x_trend, y=y_trend, mode='lines', name='Trend Line', line=dict(color='blue')))
    if show_median_line:
        median_price = vehicles_df['price'].median()
        fig.add_shape(type='line',
                      line=dict(dash='dash', color='green', width=2),
                      x0=median_price, y0=0,
                      x1=median_price, y1=1,
                      xref='x', yref='paper',
                      name='Median Price')
        fig.add_annotation(x=median_price, y=0.95, xref='x', yref='paper',
                           text='Median Price', showarrow=False, font=dict(color='green'))
    if show_mean_line:
        mean_price = vehicles_df['price'].mean()
        fig.add_shape(type='line',
                      line=dict(dash='dash', color='red', width=2),
                      x0=mean_price, y0=0,
                      x1=mean_price, y1=1,
                      xref='x', yref='paper',
                      name='Mean Price')
        fig.add_annotation(x=mean_price, y=0.90, xref='x', yref='paper',
                           text='Mean Price', showarrow=False, font=dict(color='red'))
    fig.show()


# Create interactive widget with additional controls for adjusting the number of bins and showing median and mean lines
interactive_plot = interactive(update_histogram,
                               bins=widgets.IntSlider(value=20, min=100, max=749, step=1, description='Bins'),
                               # Adjusted bin slider range for practicality
                               show_median_line=widgets.Checkbox(value=False, description='Show Median Line'),
                               show_mean_line=widgets.Checkbox(value=False, description='Show Mean Line'),
                               show_trend_line=st.sidebar.checkbox('Show Trend Line', value=False))
display(interactive_plot)

#streamlit run D:\GIthub repository\sprint-4\app.py [ARGUMENTS]
interactive(children=(Checkbox(value=False, description='Show Trend Line'), Output()), _dom_classes=('widget-interact',))
VBox(children=(Checkbox(value=False, description='Show Trend Line', layout=Layout(margin='0 0 0 20px', width='auto')),
               Checkbox(value=False, description='Remove Outliers', layout=Layout(margin='0 0 0 20px', width='auto'))))
interactive(children=(
Checkbox(value=False, description='Show Trend Line', layout=Layout(margin='0 0 0 20px', width='auto')),
Checkbox(value=False, description='Remove Outliers', layout=Layout(margin='0 0 0 20px', width='auto')), Output()),
            _dom_classes=('widget-interact',))
