import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from loguru import logger


#####################################
#Load data function
#####################################
def load_transform_welltest_data(file_path, well_name='cheetah-20'):
    """
    Load well test data from an excel sheet
    
    """
    #Load data - cheetah-20 as example
    logger.info(f"Loading data from {file_path} for well {well_name}")
    df = pd.read_excel(file_path, sheet_name=well_name)
    logger.info(f"Done loading data from {file_path} for well {well_name}")

    #Select key columns
    df = df[['Date', 'Well Name', 'WT LIQ', 'WT Oil', 'WT THP', 'WT WCT', 'Z1 BHP',
       'Z2 BHP', 'Z3 BHP', 'Delta Liquid', 'Delta Oil', 'Delta THP',
       'Delta WCT', 'Delta Z1 BHP', 'Delta Z2 BHP', 'Delta Z3BHP',
       'Decline Curve', 'Zonal Configuration', 'Engineer Interp',
       'Engineer Action', 'Notification']].copy()
    
    #Rename columns
    df.rename(columns={'Well Name':'WellName',
                        'WT LIQ':'WTLIQ',
                        'WT Oil':'WTOil',
                        'WT THP':'WTTHP',
                        'WT WCT':'WTWCT',
                        'Z1 BHP':'Z1BHP',
                        'Z2 BHP':'Z2BHP',
                        'Z3 BHP':'Z3BHP',
                        'Delta Liquid':'DeltaLiquid',
                        'Delta Oil':'DeltaOil',
                        'Delta THP':'DeltaTHP',
                        'Delta WCT':'DeltaWCT',
                        'Delta Z1 BHP':'DeltaZ1BHP',
                        'Delta Z2 BHP':'DeltaZ2BHP',
                        'Delta Z3 BHP':'DeltaZ3BHP',
                        'Decline Curve':'DeclineCurve',
                        'Zonal Configuration':'ZonalConfiguration',
                        'Engineer Interp':'EngineerInterp',
                        'Engineer Action':'EngineerAction',
                        'Notification':'Notification'
                        }, inplace=True)
    
    df.dropna(inplace = True)

    # Generate the log of changes
    # Generate the mean value of the BHP across the zones
    logger.info("Generating mean BHP and log differences")
    df['mean_bhp'] =  df[['Z1BHP', 'Z2BHP', 'Z3BHP']].mean(axis=1)
    df['log_diff_z1bhp_meanbhp'] = np.log(df['Z1BHP'] / df['mean_bhp'])
    df['log_diff_z2bhp_meanbhp'] = np.log(df['Z2BHP'] / df['mean_bhp'])
    df['log_diff_z3bhp_meanbhp'] = np.log(df['Z3BHP'] / df['mean_bhp'])

    # Generate the log of changes of other well test parameters
    df['log_diff_oil'] = np.log(df['WTOil'] / df['WTOil'].shift(1))
    df['log_diff_liq'] = np.log(df['WTLIQ'] / df['WTLIQ'].shift(1))
    df['log_diff_thp'] = np.log(df['WTTHP'] / df['WTTHP'].shift(1))
    df['log_diff_wct'] = np.log(df['WTWCT'] / df['WTWCT'].shift(1))
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