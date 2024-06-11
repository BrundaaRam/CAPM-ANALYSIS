import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.api import OLS
import statsmodels.api as sm

# Function to plot interactive chart
def interactive_plot(df):
    fig = px.line()
    for col in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[col], name=col)
    fig.update_layout(width=450, margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

# Function to normalize the prices based on the initial price
def normalize(df):
    df_normalized = df.copy()
    for col in df.columns[1:]:
        df_normalized[col] = df_normalized[col] / df_normalized[col][0]
    return df_normalized

# Function to calculate daily returns
def daily_return(df):
    df_daily_return = df.copy()
    for col in df.columns[1:]:
        df_daily_return[col] = df_daily_return[col].pct_change() * 100
    df_daily_return.fillna(0, inplace=True)
    return df_daily_return

# Function to calculate beta
def calculate_beta(stocks_daily_return, stock):
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[stock], 1)
    return b, a

# Function to calculate Sharpe Ratio
def sharpe_ratio(df, rf):
    sharpe_ratios = {}
    for col in df.columns[1:]:
        sharpe_ratios[col] = (df[col].mean() - rf) / df[col].std() * np.sqrt(252)
    return sharpe_ratios

# Function to plot scatter plot
def scatter_plot(df, stock):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sp500'], df[stock], alpha=0.5)
    plt.xlabel('SP500 Returns')
    plt.ylabel(f'{stock} Returns')
    plt.title(f'{stock} vs SP500')
    plt.grid(True)
    return plt
