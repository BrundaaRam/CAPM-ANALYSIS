import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import importlib
import CAPM_Functions
import matplotlib.pyplot as plt
import statsmodels.api as sm

importlib.reload(CAPM_Functions)

st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upwards_trend",
                   layout='wide')

st.title("Capital Asset Pricing Model")
st.subheader("Brunda R")
# Getting inputs from users
col1, col2 = st.columns([1, 1])
with col1:
    stocks_list = st.multiselect("Choose 4 stocks",
                                 ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'),
                                 ['TSLA', 'AAPL', 'AMZN', 'GOOGL'])
with col2:
    year = st.number_input("Number of years", 1, 10)

# Validate that the number of selected stocks is exactly 4
if len(stocks_list) != 4:
    st.error("Please select exactly 4 stocks.")
    st.stop()

# Specifying start and end dates for S&P 500 data
end = datetime.date.today()
start = datetime.date(end.year - year, end.month, end.day)
sp500_start = start # You can modify the start date for the S&P 500 data
sp500_end = end

# Downloading data for SP500
SP500 = web.DataReader(['sp500'], 'fred', sp500_start, sp500_end)

stocks_df = pd.DataFrame()
for stock in stocks_list:
    data = yf.download(stock, start=start, end=end)
    stocks_df[f'{stock}'] = data['Close']

stocks_df.reset_index(inplace=True)
SP500.reset_index(inplace=True)
SP500.columns = ['Date', 'sp500']
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### DataFrame head")
    st.dataframe(stocks_df.head(), use_container_width=True)
with col2:
    st.markdown("### DataFrame tail")
    st.dataframe(stocks_df.tail(), use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Price of all the Stocks")
    st.plotly_chart(CAPM_Functions.interactive_plot(stocks_df))
with col2:
    normalized_df = CAPM_Functions.normalize(stocks_df)
    st.markdown("### Price of all the Stocks (After Normalizing)")
    st.plotly_chart(CAPM_Functions.interactive_plot(normalized_df))

stocks_daily_return = CAPM_Functions.daily_return(stocks_df)
st.write(stocks_daily_return.head())

# Save to CSV in-memory
csv = stocks_daily_return.to_csv(index=False)
st.download_button(
    label="Download Daily Returns CSV",
    data=csv,
    file_name='stocks_daily_return.csv',
    mime='text/csv'
)

# Risk-Free Rate Selection
rf_options = {'US 10-Year Treasury': 0.025, '3-Month Treasury Bill': 0.005}
rf_selected = st.selectbox("Select Risk-Free Rate", list(rf_options.keys()))
rf = rf_options[rf_selected]

# Calculating Beta and Alpha
beta = {}
alpha = {}
for i in stocks_daily_return.columns:
    if i != 'Date' and i != 'sp500':
        b, a = CAPM_Functions.calculate_beta(stocks_daily_return, i)
        beta[i] = b
        alpha[i] = a

st.write("Beta Values:", beta)
st.write("Alpha Values:", alpha)

# Display Beta Values
beta_df = pd.DataFrame(list(beta.items()), columns=['Stock', 'Beta Value'])
beta_df['Beta Value'] = beta_df['Beta Value'].round(2)
st.markdown('### Calculated Beta Value')
st.dataframe(beta_df, use_container_width=True)

# Calculating Expected Return using CAPM
rm = stocks_daily_return['sp500'].mean() * 252
return_df = pd.DataFrame({
    'Stock': stocks_list,
    'Return Value': [round(rf + beta[stock] * (rm - rf), 2) for stock in stocks_list]
})
st.markdown('### Calculated Return using CAPM')
st.dataframe(return_df, use_container_width=True)

# Portfolio Analysis
portfolio_weights = [0.25, 0.25, 0.25, 0.25]  # Adjust weights as necessary
portfolio_beta = sum(beta[stock] * weight for stock, weight in zip(stocks_list, portfolio_weights))
portfolio_expected_return = rf + portfolio_beta * (rm - rf)
st.write(f"Portfolio Beta: {portfolio_beta:.2f}")
st.write(f"Portfolio Expected Return: {portfolio_expected_return:.2f}%")

# Sharpe Ratios
sharpe_ratios = CAPM_Functions.sharpe_ratio(stocks_daily_return, rf)
st.write("Sharpe Ratios:", sharpe_ratios)

# Scatter Plots for visualization
for stock in stocks_list:
    st.write(f'{stock} vs SP500 Scatter Plot')
    st.pyplot(CAPM_Functions.scatter_plot(stocks_daily_return, stock))
