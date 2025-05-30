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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

app = Flask(__name__, template_folder='templates')

def fetch_ticker_news(ticker, total_headlines=150):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    driver.get(url)

    headlines = set()
    scroll_pause_time = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while len(headlines) < total_headlines:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        
        elems = driver.find_elements(By.CSS_SELECTOR, 'h3.clamp')
        for elem in elems:
            text = elem.text.strip()
            if text:
                headlines.add(text)
            if len(headlines) >= total_headlines:
                break
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.quit()
    return list(headlines)[:total_headlines]

def get_sentiment_score(headlines):
    if not headlines or headlines == ["No news found"]:
        return 0.0
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = probs[:, 2] - probs[:, 0]  # positive - negative
    avg_score = scores.mean().item()
    return avg_score

def fetch_stock_data(ticker, days=15):
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=days*2)  # buffer for weekends/holidays
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
    if df.empty or 'Close' not in df.columns:
        return None
    df = df.reset_index()[['Date', 'Close']]
    df = df.tail(days).reset_index(drop=True)
    if len(df) < days:
        return None
    return df

def prepare_dataset(ticker):
    days = 15
    headlines_per_day = 10
    total_headlines = days * headlines_per_day
    
    stock_df = fetch_stock_data(ticker, days=days)
    if stock_df is None or stock_df.empty:
        return None

    headlines = fetch_ticker_news(ticker, total_headlines=total_headlines)
    if headlines == ["No news found"] or len(headlines) < total_headlines:
        return None

    daily_sentiments = []
    for day_idx in range(days):
        start_idx = day_idx * headlines_per_day
        end_idx = start_idx + headlines_per_day
        daily_chunk = headlines[start_idx:end_idx]
        sentiment = get_sentiment_score(daily_chunk)
        daily_sentiments.append(sentiment)

    stock_df['Sentiment_Score'] = daily_sentiments
    return stock_df

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
    data_df = prepare_dataset(ticker)

    if data_df is None or data_df.empty or len(data_df) < 4:
        return jsonify({'error': 'Not enough stock or sentiment data for this ticker.'}), 400

    features = data_df[['Close', 'Sentiment_Score']].values
    X, y = create_sequences(features, window_size=3)
    if len(X) == 0:
        return jsonify({'error': 'Not enough data for sequence creation.'}), 400

    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    y = y[:, 0]  # predict next day close price

    try:
        trained_model = build_and_train_model(X, y)
        prediction = trained_model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))[0][0]
    except Exception as e:
        return jsonify({'error': f'Model training or prediction failed: {e}'}), 500

    latest_close = float(data_df['Close'].iloc[-1])
    signal = "Buy" if prediction > latest_close else "Sell"

    # Prepare data for the table: convert dates to strings
    data_table = []
    for _, row in data_df.iterrows():
        date_val = row['Date']
        if isinstance(date_val, pd.Series):
            date_val = date_val.iloc[0]  # get scalar timestamp
        data_table.append({
            'date': date_val.strftime('%Y-%m-%d'),
            'close': round(float(row['Close']), 2),
            'sentiment': round(float(row['Sentiment_Score']), 4)
        })

    return jsonify({
        'ticker': ticker,
        'prediction': round(float(prediction), 4),
        'signal': signal,
        'table_data': data_table
    })

if __name__ == '__main__':
    app.run(debug=True)
