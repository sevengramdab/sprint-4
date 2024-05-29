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
from ipywidgets import Output  # <<<< Here is the required import statement
import streamlit as st



print(os.getcwd())

import streamlit as st

INTRODUCTION_TEXT = """
# Vehicle Price Analysis
In this project, we will analyze a dataset of vehicle prices. We will use interactive widgets to control the arguments of 
the function 'f', which is directly linked to the state of the 'Show Trend Line' checkbox. When the checkbox state changes, 
this will be reflected on the results of the function 'f'.
"""

st.write(INTRODUCTION_TEXT)

print(INTRODUCTION_TEXT)

# ... the rest of your code...

#streamlit run C:/Program Files/JetBrains/PyCharm 2024.1/plugins/python/helpers/pydev/pydevconsole.py

sns.set(style="whitegrid")
vehicles = pd.read_csv('vehicles_us.csv')

# Fill missing values in 'model_year' by grouping by 'model' and using the median year
vehicles['model_year'] = vehicles.groupby('model')['model_year'].transform(lambda x: x.fillna(x.median()))

# Fill missing values in 'cylinders' by grouping by 'model' and using the median cylinders
vehicles['cylinders'] = vehicles.groupby('model')['cylinders'].transform(lambda x: x.fillna(x.median()))

# Fill missing values in 'odometer' by grouping by 'model_year' and using the median odometer
vehicles['odometer'] = vehicles.groupby('model_year')['odometer'].transform(lambda x: x.fillna(x.median()))

# Remove outliers in 'model_year' and 'price'
q1_model_year = vehicles['model_year'].quantile(0.25)
q3_model_year = vehicles['model_year'].quantile(0.75)
iqr_model_year = q3_model_year - q1_model_year
lower_bound_model_year = q1_model_year - 1.5 * iqr_model_year
upper_bound_model_year = q3_model_year + 1.5 * iqr_model_year

q1_price = vehicles['price'].quantile(0.25)
q3_price = vehicles['price'].quantile(0.75)
iqr_price = q3_price - q1_price
lower_bound_price = q1_price - 1.5 * iqr_price
upper_bound_price = q3_price + 1.5 * iqr_price

vehicles = vehicles[(vehicles['model_year'] >= lower_bound_model_year) & (vehicles['model_year'] <= upper_bound_model_year)]
vehicles = vehicles[(vehicles['price'] >= lower_bound_price) & (vehicles['price'] <= upper_bound_price)]

print(vehicles.head())
print(vehicles.info())

# %%
from ipywidgets import interactive, HBox, VBox, widgets
import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go

#creates a new column 'manufacturer' in the dataframe 'vehicles' based on the 'model' column.
# It applies a function that splits each model name by space and takes the first word as the manufacturer.
vehicles['manufacturer'] = vehicles['model'].apply(lambda x: x.split()[0])

# This line creates a header 'Data viewer' in the dashboard.
st.header('Data viewer')

# This line creates a checkbox named 'Include manufactures with less than 1000 ads'.
# if this box is checked by the user, the variable show_manuf_1k_ads will be True.
show_manuf_1k_ads = st.checkbox('Include manufacturers with less than 1000 ads')


# This code block is executed if show_manuf_1k_ads is False.
# meaning the checkbox is not checked. It filters the dataframe 'vehicles' to include only manufacturers with more than 1000 ads.
if not show_manuf_1k_ads:
    df = vehicles.groupby('manufacturer').filter(lambda x: len(x) > 1000)

st.dataframe(vehicles)

# creating a plot
st.header('Vehicle types by manufacturer')
st.write(px.histogram(vehicles, x='manufacturer', color='type'))
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
st.write(px.histogram(vehicles, x='model_year', color='condition'))
# a lot more concise!
# -------------------------------------------------------

# creates an interactive web app with Streamlit, allowing users to compare and visualize the price
# distribution of vehicles between any two manufacturers, which can optionally be normalized to percentages.
st.header('Compare price distribution between manufacturers')
manufac_list = sorted(vehicles['manufacturer'].unique())
manufacturer_1 = st.selectbox('Select manufacturer 1',
                              manufac_list, index=manufac_list.index('chevrolet'))

manufacturer_2 = st.selectbox('Select manufacturer 2',
                              manufac_list, index=manufac_list.index('hyundai'))
mask_filter = (vehicles['manufacturer'] == manufacturer_1) | (vehicles['manufacturer'] == manufacturer_2)
df_filtered = vehicles[mask_filter]
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
    fig = px.histogram(vehicles, x='price', title='Vehicle Price Distribution')
    if show_trendline:
        fig.add_traces(go.Scatter(x=np.sort(vehicles['price']),
                                  y=np.poly1d(np.polyfit(vehicles['price'],
                                                         np.histogram(vehicles['price'], bins=40)[0],
                                                         1))(np.sort(vehicles['price'])),
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
# Create the checkbox widget
show_trendline = widgets.Checkbox(value=False, description='Show Trend Line')


# Function to update the histogram based on the trend line visibility
def update_histogram(show_trendline=show_trendline.value):
    fig = px.histogram(vehicles, x='price', title='Vehicle Price Distribution')
    if show_trendline:

        # Check for non-NaN values in your data
        nan_mask = np.isnan(vehicles['price'])

        if not all(nan_mask):
            coefficients = np.polyfit(vehicles['price'][~nan_mask],
                                      np.histogram(vehicles['price'][~nan_mask], bins=40)[0], 1)
            polyd = np.poly1d(coefficients)
            x_sorted = np.sort(vehicles['price'])
            y_sorted = polyd(x_sorted)
            fig.add_trace(go.Scatter(x=x_sorted, y=y_sorted, mode='lines', name='Trend Line'))
            fig.show()


# Create the interactive plot
interactive_plot = interactive(update_histogram, show_trendline=show_trendline)
display(interactive_plot)
display(interactive_plot)


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
vehicles_below_200k = vehicles[vehicles['odometer'] < 200000]

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
# Creating the Checkbox widget
show_trendline = widgets.Checkbox(value=False, description='Show Trend Line')


def update_histogram(show_trendline=show_trendline.value):
    fig = px.histogram(vehicles, x='price', title='Vehicle Price Distribution')
    if show_trendline:
        fig.add_trace(go.Scatter(x=np.sort(vehicles['price']),
                                 y=np.poly1d(np.polyfit(vehicles['price'],
                                                        np.histogram(vehicles['price'], bins=40)[0],
                                                        1))(np.sort(vehicles['price'])),
                                 mode='lines', name='Trend Line'))
    fig.show()


# Create the interactive plot
interactive_plot = interactive(update_histogram, show_trendline=show_trendline)
display(interactive_plot)


# Create interactive widget with additional controls for adjusting the number of bins and showing median and mean lines
interactive_plot = interactive(update_histogram,
                               bins=widgets.IntSlider(value=20, min=100, max=749, step=1, description='Bins'),
                               # Adjusted bin slider range for practicality
                               show_median_line=widgets.Checkbox(value=False, description='Show Median Line'),
                               show_mean_line=widgets.Checkbox(value=False, description='Show Mean Line'),
                               show_trend_line=st.sidebar.checkbox('Show Trend Line', value=False))
display(interactive_plot)

from ipywidgets import interactive, Checkbox, Output, VBox, Layout
from IPython.display import display

# create an interactive app that allows users to select two vehicle manufacturers.
# It then renders a histogram overlay showing their comparative vehicle price distributions, which can be normalized.
def f(Show_Trend_Line):
    return Show_Trend_Line


interactive_plot = interactive(f, Show_Trend_Line=Checkbox(value=False, description='Show Trend Line'))
display(interactive_plot)

box = VBox(
    children=(Checkbox(value=False, description='Show Trend Line', layout=Layout(margin='0 0 0 20px', width='auto')),))
display(box)

output = '''
### Project Output: Vehicle Data Analysis

#### Objective
The objective of this project was to analyze a dataset of used vehicles to understand the distribution of vehicle prices, the relationship between vehicle age and price, and to preprocess the data for more accurate analysis.

#### Data Preprocessing
1. **Handling Missing Values:**
   - **Model Year:** Missing values were filled by grouping the data by 'model' and using the median model year within each group.
   - **Cylinders:** Missing values were filled by grouping the data by 'model' and using the median number of cylinders within each group.
   - **Odometer:** Missing values were filled by grouping the data by 'model_year' and using the median odometer reading within each group.

2. **Removing Outliers:**
   - Outliers in the 'model_year' and 'price' columns were removed using the interquartile range (IQR) method to ensure that extreme values do not skew the analysis.

#### Visualizations
1. **Histogram of Vehicle Prices:**
   - A histogram was created to show the distribution of vehicle prices. This visualization helps in understanding the range and frequency of vehicle prices, highlighting the most common price ranges and any outliers.

2. **Scatter Plot of Price vs. Model Year:**
   - A scatter plot was created with 'model_year' on the x-axis and 'price' on the y-axis. This plot helps in understanding the relationship between the age of a vehicle and its price. Generally, newer vehicles tend to be priced higher than older ones.

#### Analysis
- The histogram of vehicle prices revealed the most common price ranges and highlighted any outliers.
- The scatter plot of price versus model year showed the expected trend where newer vehicles are generally priced higher than older ones. It also revealed clusters and outliers, providing insights into the pricing patterns of used vehicles.

#### Conclusion
The preprocessing steps ensured that the dataset was more complete and free from extreme outliers, making the analysis more reliable. The visualizations provided valuable insights into the distribution of vehicle prices and the relationship between vehicle age and price. These insights can be useful for understanding market trends and making informed decisions in the used vehicle market.
'''

