import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from loguru import logger
import streamlit as st


#####################################
#Load data function
#####################################
@st.cache_data()
def load_transform_welltest_data(file_path, well_name='cheetah-20', threshold=0.1):
    """
    Load well test data from an excel sheet
    
    """
    #Load data - cheetah-20 as example
    logger.info(f"Loading data from {file_path} for well {well_name}")
    df = pd.read_excel(file_path, sheet_name=well_name)
    logger.info(f"Done loading data from {file_path} for well {well_name}")

    if "WellName" in df.columns:
        df.rename(columns={'WellName': 'Well Name'}, inplace=True)
    elif "Well Name" in df.columns:
        # If the column is already named 'Well Name', do nothing
        pass
    else:
        logger.error("The DataFrame does not contain a 'WellName' or 'Well Name' column.")
        raise ValueError("The DataFrame does not contain a 'WellName' or 'Well Name' column.")
    #Select key columns
    df = df[['Date', 'Well Name', 'WT LIQ', 'WT Oil', 'WT THP', 'WT WCT', 'Z1 BHP',
       'Z2 BHP', 'Z3 BHP']].copy()
    
    #Rename columns
    df.rename(columns={'Well Name':'WellName',
                        'WT LIQ':'WTLIQ',
                        'WT Oil':'WTOil',
                        'WT THP':'WTTHP',
                        'WT WCT':'WTWCT',
                        'Z1 BHP':'Z1BHP',
                        'Z2 BHP':'Z2BHP',
                        'Z3 BHP':'Z3BHP',
                        }, inplace=True)
    

    #df.dropna(inplace = True)
    df['WTWCT'] = df['WTWCT']/ 100  # Convert WCT from percentage to decimal

    # Generate the log of changes
    # Generate the mean value of the BHP across the zones
    logger.info("Generating mean BHP and log differences")
    df['mean_bhp'] =  df[['Z1BHP', 'Z2BHP', 'Z3BHP']].mean(axis=1)
    df['log_diff_z1bhp_meanbhp'] = np.log(df['Z1BHP'] / df['mean_bhp'])
    df['log_diff_z2bhp_meanbhp'] = np.log(df['Z2BHP'] / df['mean_bhp'])
    df['log_diff_z3bhp_meanbhp'] = np.log(df['Z3BHP'] / df['mean_bhp'])
    df['zone1_status'] = np.where(df['log_diff_z1bhp_meanbhp'] > threshold, 'Closed', 'Open')
    df['zone2_status'] = np.where(df['log_diff_z2bhp_meanbhp'] > threshold, 'Closed', 'Open')
    df['zone3_status'] = np.where(df['log_diff_z3bhp_meanbhp'] > threshold, 'Closed', 'Open')

    # Generate the log of changes of other well test parameters
    df['log_diff_oil'] = np.log(df['WTOil'] / df['WTOil'].shift(1))
    df['log_diff_liq'] = np.log(df['WTLIQ'] / df['WTLIQ'].shift(1))
    df['log_diff_thp'] = np.log(df['WTTHP'] / df['WTTHP'].shift(1))
    df['log_diff_wct'] = np.log(df['WTWCT'] / df['WTWCT'].shift(1))
    df['log_diff_z3bhp'] = np.log(df['Z3BHP'] / df['Z3BHP'].shift(1))
    df['log_diff_z2bhp'] = np.log(df['Z2BHP'] / df['Z2BHP'].shift(1))
    df['log_diff_z1bhp'] = np.log(df['Z1BHP'] / df['Z1BHP'].shift(1))
    logger.info("Done generating mean BHP and log differences")

    #print(df.head(3))

    return df


#####################################
#Generating plots
#####################################
def make_plot(data, x, y, title, x_label, y_label, kind='line', color=None):
    if kind == 'line':
        fig = px.line(data, x=x, y=y, height=600, width = 1000,markers=True, title=title)
    elif kind == 'scatter':
        fig = px.scatter(data, x=x, y=y, height=600, width = 1000, title=title)
    elif kind == 'bar':
        fig = px.bar(data, x=x, y=y, height=600, width = 1000, title=title)
    else:
        raise ValueError("Unsupported plot type. Use 'line', 'scatter', or 'bar'.")
    
    if color:
        fig.update_traces(marker_color=color)
    
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    fig.show()
    fig.write_html(f"charts/{title}.html")

def make_plot_2y(data, x, y, y2, title, x_label, y_label, kind='line', color=None):
    if kind == 'line':
        fig = px.line(data, x=x, y=y, height=600, width = 1000,markers=True, title=title)
    elif kind == 'scatter':
        fig = px.scatter(data, x=x, y=y, height=600, width = 1000, title=title)
    else:
        raise ValueError("Unsupported plot type. Use 'line', 'scatter'.")

    if fig:
        fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[y2],
                    mode='lines+markers',
                    name=y2,
                    )
        )
        # Update layout to include multiple y-axes
        fig.update_layout(
            yaxis2=dict(
                title='Y-Axis 2',
                overlaying='y',
                side='right'
                )
        )
        
        if color:
            fig.update_traces(marker_color=color)
        
        fig.show()

        #Save file in the charts folder in html format  
        fig.write_html(f"charts/{title}.html")


#####################################
#Decline curve analysis
#####################################
# Function to fit the decline curve
def fit_decline_curve(data, time_col, rate_col, auto = True, qi = None, Di = None):
    # Extract time and rate data
    t = data[time_col]
    q = data[rate_col]

    # Define the exponential decline function
    def exponential_decline(t, qi, Di):
        return qi * np.exp(-Di * t)
    
    # Fit the exponential decline curve
    popt, _ = curve_fit(exponential_decline, t, q, maxfev=10000)
    
    if auto == True:
        # Extract fitted parameters
        qi, Di = popt
    else: #Define the parameters manually
        qi = 6000
        Di = 0.008
    
    # Generate fitted values
    q_fit = exponential_decline(t, qi, Di)
    
    # Return fitted parameters and values
    return qi, Di, q_fit

#Clean the dataset by removing outliers in oil rate iteratively

#Fit hyperbolic decline curve
def fit_hyperbolic_decline_curve(data, time_col, rate_col, auto=True, qi=None, Di=None, b=None):
    # Extract time and rate data

    if len(data) >= 5:
        #only use the top 5 rows for fitting
        t = data[time_col].head(5)
        q = data[rate_col].head(5)
    elif len(data) < 3:
        logger.error("Not enough data points to fit the hyperbolic decline curve. At least 3 data points are required.")
        return None, None
    else:
        t = data[time_col]
        q = data[rate_col]

    # Define the hyperbolic decline function
    def hyperbolic_decline(t, qi, b, Di):
        return qi / ((1 + b * Di * t) ** (1 / b))
    
    #initialize parameters if not provided
    qi_initial = 8000
    b_initial = 0.5
    Di_initial = 0.05

    #bounds for the parameters
    qi_max = 9000
    b_max = 1
    Di_max = 0.2
    
    # Fit the hyperbolic decline curve
    popt, _ = curve_fit(hyperbolic_decline, t, q, 
                        p0=[qi_initial, b_initial, Di_initial],
                        bounds=(0, [qi_max, b_max, Di_max]),
                        maxfev=100000)
        
    # Generate fitted values
    q_fit = hyperbolic_decline(t, *popt)
    
    # Return fitted parameters and values
    return q_fit, popt

def dca_forecast(data, time_col, popt):
    """
    """
    # Define the hyperbolic decline function
    def hyperbolic_decline(t, qi, b, Di):
        return qi / ((1 + b * Di * t) ** (1 / b))
    # Generate forecasted values
    data[time_col] = (data['Date'] - data['Date'].min()).dt.days  # Convert Date to days since the first date
    t = data[time_col]
    q_forecast = hyperbolic_decline(t, *popt)
    data['dca_rate'] = q_forecast
    return data




def remove_outliers(df_subset, threshold=0.4):
    """
    Remove outliers from the dataset based on the specified threshold.
    """
    df_subset = df_subset.copy()
    clean_rate = [df_subset['WT Oil'].iloc[0]]
    index_list = [df_subset.index[0]]
    
    for index, row in df_subset.iterrows():
        diff = np.log(row['WT Oil']/clean_rate[-1])
        if diff > -0.4:
            clean_rate.append(row['WT Oil'])
            index_list.append(index)

    return df_subset, index_list