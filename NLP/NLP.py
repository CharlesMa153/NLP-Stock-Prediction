from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

app = Flask(__name__, template_folder='templates')

def fetch_ticker_news(ticker, max_headlines=500):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    #these options are used to run the browser in headless mode, so no GUI is displayed but it is searching through the webpage
    driver = webdriver.Chrome(options=options)

    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    driver.get(url)

    headlines = [] #append is used for list
    seen = set() #add is used for sets
    scroll_pause_time = 2
    last_height = driver.execute_script("return document.body.scrollHeight")

    while len(headlines) < max_headlines:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        elems = driver.find_elements(By.CSS_SELECTOR, 'h3.clamp')
        for elem in elems:
            text = elem.text.strip() #cleans the text so there is no white space
            if text and text not in seen:
                headlines.append(text)
                seen.add(text)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.quit()
    return headlines

def assign_headlines_to_days(headlines, num_days):
    total = len(headlines)
    base = total // num_days #// rounds to closest integer
    remainder = total % num_days

    daily_chunks = []
    start_idx = 0

    for i in range(num_days):
        count = base + (1 if i < remainder else 0)
        daily_chunks.append(headlines[start_idx:start_idx + count])
        start_idx += count

    return daily_chunks

def get_sentiment_score(headlines):
    if not headlines or headlines == ["No news found"]:
        return 0.0
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt") #tokenizes the headlines and prepares them for the model
    with torch.no_grad(): #tells model that we are not training but only testing
        outputs = model(**inputs) #returns logits of the model
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1) #softmax is used to convert logits to probabilities
    scores = probs[:, 2] - probs[:, 0]  # positive - negative
    avg_score = scores.mean().item() #.item() converts the PyTorch tensor to a Python float
    return avg_score

def fetch_stock_data(ticker, days=15):
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=days*2) #for weekends and holidays, so we get double the data just in case
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
    if df.empty or 'Close' not in df.columns:
        return None
    df = df.reset_index()[['Date', 'Close']] #make index (date) into a column and select only Date and Close columns
    #[[]] returns dataframe, [] returns series
    df = df.tail(days).reset_index(drop=True) #tail(days) gets the first 15 rows, and drops useless index column that is beyond 15
    if len(df) < days:
        return None
    return df

def prepare_dataset(ticker):
    days = 15

    stock_df = fetch_stock_data(ticker, days=days)
    if stock_df is None or stock_df.empty:
        return None, 0

    headlines = fetch_ticker_news(ticker, max_headlines=500)
    if not headlines or len(headlines) < days:
        return None, len(headlines)

    daily_chunks = assign_headlines_to_days(headlines, days)

    daily_sentiments = []
    for chunk in daily_chunks:
        sentiment = get_sentiment_score(chunk)
        daily_sentiments.append(sentiment)

    stock_df['Sentiment_Score'] = daily_sentiments[::-1] #reverse to match stock news order, now it starts from the most recent date
    return stock_df, len(headlines)

def create_sequences(data, window_size=3):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_and_train_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=1, validation_split=0.2, verbose=0)
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'ticker' not in data:
        return jsonify({'error': 'Ticker symbol required'}), 400

    ticker = data['ticker'].upper()
    data_df, headline_count = prepare_dataset(ticker)

    if data_df is None or data_df.empty or len(data_df) < 4:
        return jsonify({'error': 'Not enough stock or sentiment data for this ticker.'}), 400

    features = data_df[['Close', 'Sentiment_Score']].values

    #scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    X, y = create_sequences(features_scaled, window_size=3)
    if len(X) == 0:
        return jsonify({'error': 'Not enough data for sequence creation.'}), 400

    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    y = y[:, 0]  #predict Close price only (scaled)

    try:
        trained_model = build_and_train_model(X, y)
        
        #predict for each sequence to get predictions for past days
        preds_scaled = trained_model.predict(X).flatten()  #all predictions scaled
        
        #invert scaling for all predicted Close prices
        close_min = scaler.data_min_[0]
        close_max = scaler.data_max_[0]
        preds = preds_scaled * (close_max - close_min) + close_min
        
    except Exception as e:
        return jsonify({'error': f'Model training or prediction failed: {e}'}), 500

    #prediction for next day (last available sequence)
    pred_scaled = preds_scaled[-1]
    prediction = preds[-1]
    latest_close = float(data_df['Close'].iloc[-1])
    signal = "Buy" if prediction > latest_close else "Sell"

    #prepare table data with difference column
    #because sequences start after window_size days, predictions are offset by window_size
    window_size = 3
    data_table = []
    for i, row in data_df.iterrows():
        date_val = row['Date']
        if isinstance(date_val, pd.Series):
            date_val = date_val.iloc[0]
        
        actual_close = float(row['Close'])
        
        #for first 'window_size' days, no prediction (None)
        if i < window_size:
            diff = None
            pred_price = None
        else:
            pred_price = preds[i - window_size]
            diff = round(actual_close - pred_price, 4)
        
        data_table.append({
            'date': date_val.strftime('%Y-%m-%d'),
            'close': round(actual_close, 2),
            'sentiment': round(float(row['Sentiment_Score']), 4),
            'predicted_close': round(pred_price, 4) if pred_price is not None else None,
            'difference': diff
        })

    return jsonify({
        'ticker': ticker,
        'prediction': round(float(prediction), 4),
        'signal': signal,
        'headline_count': headline_count,
        'table_data': data_table
    })


if __name__ == '__main__':
    app.run(debug=True)
