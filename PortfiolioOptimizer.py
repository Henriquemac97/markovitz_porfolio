import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
import datetime
from pandas_datareader import data as pdr

def dataframe(wallet, data_source='yahoo', start, end):

    df = pd.DataFrame()
    for i in list(wallet.keys()):
        df[i] = pdr.DataReader(i,
                               data_source=data_source,
                               start=start,
                               end=end)["Adj Close"]

    return df

def time_series_plot(df):
    fig = px.line(df, x=df.reset_index()['Date'], y=df.columns)

    fig.update_xaxes(title_text='Time', dtick="M1", tickformat="%b\n%Y")

    fig.update_yaxes(title_text='Price')

    fig.update_layout(
        dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))

    fig.update_layout(legend=dict(title='Stocks'))

    fig.update_layout(title='Stocks prices over time')

    fig.update_xaxes(title_text='Time')

    fig.show()

def correlation_plot(df,method='pearson'):

    fig = px.imshow(df.corr(method),
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    range_color=[-1, 1])
    fig.update_layout(title='Correlation matrix',coloraxis_colorbar=dict(title="R2",))
    fig.show()

def generate_wallets(df, num_portfolios=10000, Rf=0):

    # vetores de dados
    portfolio_weights = []
    portfolio_exp_returns = []
    portfolio_vol = []
    portfolio_sharpe = []

    # retorno simples
    r = df.pct_change()
    mean_returns = r.mean() * df.shape[0]

    # matriz de covariância
    covariance = np.cov(r[1:].T)

    for i in range(num_portfolios):
        # gerando pesos aleatórios
        k = np.random.rand(len(df.columns))
        w = k / sum(k)

        # retorno
        R = np.dot(mean_returns, w)

        # risco
        vol = np.sqrt(np.dot(w.T, np.dot(covariance, w))) * np.sqrt(
            df.shape[0])

        # sharpe ratio
        sharpe = (R - Rf) / vol

        portfolio_weights.append(w)
        portfolio_exp_returns.append(R)
        portfolio_vol.append(vol)
        portfolio_sharpe.append(sharpe)

    wallets = {
        'weights': portfolio_weights,
        'return': portfolio_exp_returns,
        'risk': portfolio_vol,
        'sharpe ratio': portfolio_sharpe
    }

    return wallets

def plot_efficient_frontier(wallets, method):

    if method == 'risk':
        best_index = np.array(wallets[method]).argmin()
    elif method == 'return' or method == 'sharpe ratio':
        best_index = np.array(wallets[method]).argmax()

    y_axis = wallets['return'][best_index]
    X_axis = wallets['risk'][best_index]

    fig = go.Figure(data=px.scatter(
        wallets,
        x='risk',
        y='return',
        color=method,
        range_color=[min(wallets[method]),
                     max(wallets[method])],
        color_continuous_scale=px.colors.sequential.Viridis,
        opacity=0.8))

    fig.add_trace(
        go.Scatter(mode='markers',
                   x=[X_axis],
                   y=[y_axis],
                   hoverinfo='skip',
                   marker=dict(color='Red', size=10),
                   showlegend=False))

    fig.update_layout(
        dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))

    fig.update_layout(title='Efficient Frontier')

    d = fig.to_dict()
    d["data"][0]["type"] = "scatter"
    fig = go.Figure(d)
    fig.show()

def best_portfolio(wallets, method='sharpe_ratio'):

    weights = wallets['weights']

    if method == 'sharpe ratio':

        indice = np.array(wallets['sharpe ratio']).argmax()

    elif method == 'risk':

        indice = np.array(wallets['risk']).argmin()

    elif method == 'return':

        indice = np.array(wallets['return']).argmax()

    return weights[indice]

#st.title('Portfolio Optimizer')

#df = dataframe(wallet, data_source='yahoo', start, end)

#time_series_plot(df)

#correlation_plot(df,method='pearson')

#wallets = generate_wallets(df, num_portfolios=10000, Rf=0)

#plot_efficient_frontier(wallets, method)