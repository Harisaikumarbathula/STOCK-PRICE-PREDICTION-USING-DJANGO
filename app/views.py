from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt

from .models import Harisai

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm

# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    tickers = ['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM']
    data = yf.download(
        tickers=tickers,
        period='1mo',
        interval='1d',
        group_by='ticker',
        threads=True,
    )

    # Extract Close prices for each ticker
    closes = data.xs('Close', level=1, axis=1).reset_index()

    fig_left = go.Figure()
    for ticker in tickers:
        fig_left.add_trace(
            go.Scatter(x=closes['Date'], y=closes[ticker], name=ticker)
        )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')

    # ================================================ To show recent stocks ==============================================
    recent_tickers = ['AAPL', 'AMZN', 'GOOGL', 'UBER', 'TSLA']
    dfs = []
    for t in recent_tickers:
        df = yf.download(tickers=t, period='1d', interval='1d')
        if not df.empty:
            df['Ticker'] = t
            dfs.append(df)

    if dfs:
        df = pd.concat(dfs)
        df.reset_index(inplace=True)
        df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = df['Date'].astype(str)
        recent_stocks = df.to_dict('records')
    else:
        recent_stocks = []

    # ========================================== Page Render section =====================================================
    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient='records')
    ticker_list = json.loads(json_ticker)

    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })

# The Predict Function
def predict(request, ticker_value, number_of_days):
    # Validate days
    try:
        number_of_days = int(number_of_days)
        if number_of_days < 0:
            return render(request, 'Negative_Days.html', {})
        if number_of_days > 365:
            return render(request, 'Overflow_days.html', {})
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    # Validate ticker and get data
    try:
        df = yf.download(tickers=ticker_value, period='1d', interval='1m')
        if df.empty:
            raise ValueError("No data available")
    except:
        return render(request, 'API_Down.html', {})

    # ================================================= Plotting ========================================================
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='market data'
    ))
    fig.update_layout(
        title=f'{ticker_value} Live Share Price',
        yaxis_title='Stock Price (USD)',
        xaxis_rangeslider_visible=True,
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b", 
        font_color="white"
    )
    plot_div = plot(fig, auto_open=False, output_type='div')

    # ========================================== Machine Learning ========================================================
    try:
        df_ml = yf.download(ticker_value, period='3mo', interval='1h')
        if df_ml.empty:
            raise ValueError("No data available")
    except:
        return render(request, 'API_Down.html', {})

    df_ml = df_ml[['Close']]
    forecast_out = int(number_of_days)
    
    # Prepare data
    df_ml['Prediction'] = df_ml['Close'].shift(-forecast_out)
    X = np.array(df_ml.drop(['Prediction'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'].dropna())

    # Split and train
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    forecast = clf.predict(X_forecast).tolist()

    # ========================================== Prediction Plot =========================================================
    pred_dates = [dt.datetime.today() + dt.timedelta(days=i) for i in range(forecast_out)]
    pred_fig = go.Figure([go.Scatter(x=pred_dates, y=forecast)])
    pred_fig.update_layout(
        title='Price Forecast',
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Ticker Info =============================================================
    try:
        ticker_info = yf.Ticker(ticker_value).info
        info_fields = {
            'Symbol': ticker_info.get('symbol', 'N/A'),
            'Name': ticker_info.get('longName', 'N/A'),
            'Sector': ticker_info.get('sector', 'N/A'),
            'Industry': ticker_info.get('industry', 'N/A'),
            'Market Cap': ticker_info.get('marketCap', 'N/A'),
            'Country': ticker_info.get('country', 'N/A')
        }
    except:
        info_fields = {field: 'N/A' for field in ['Symbol', 'Name', 'Sector', 'Industry', 'Market Cap', 'Country']}

    return render(request, "result.html", {
        'plot_div': plot_div,
        'confidence': confidence,
        'forecast': forecast,
        'ticker_value': ticker_value,
        'number_of_days': number_of_days,
        'plot_div_pred': plot_div_pred,
        **info_fields
    })