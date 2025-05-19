import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT once
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

app = Flask(__name__, template_folder='templates')

def fetch_ticker_news(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Grab up to 20 headlines
    headlines = [h3.get_text(strip=True) for h3 in soup.select('h3.clamp.yf-1y7058a')][:20]
    return headlines if headlines else ["No news found"]

def get_sentiment_score(headlines):
    if not headlines or headlines == ["No news found"]:
        return 0.0

    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # FinBERT classes: [Negative, Neutral, Positive]
    # We'll calculate sentiment as Positive probability minus Negative probability
    scores = probs[:, 2] - probs[:, 0]  # positive - negative
    avg_score = scores.mean().item()
    return avg_score

def fetch_stock_data(ticker):
    end = date.today() + timedelta(days=1)  # Include today if available
    start = date.today() - timedelta(days=5)  # last 5 calendar days of stock data
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
    if df.empty or 'Close' not in df.columns:
        return None
    df = df.reset_index()[['Date', 'Close']]
    return df

def prepare_dataset(ticker):
    stock_df = fetch_stock_data(ticker)
    if stock_df is None or stock_df.empty:
        return None

    # Get sentiment for latest 20 headlines (once)
    headlines = fetch_ticker_news(ticker)
    sentiment_score = get_sentiment_score(headlines)

    # Assign the same sentiment score to all stock data dates (since headlines are not timestamped)
    stock_df['Sentiment_Score'] = sentiment_score
    return stock_df

def create_sequences(data, window_size=1):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_and_train_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
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
    if data_df is None or len(data_df) < 2:
        headlines = fetch_ticker_news(ticker)
        return jsonify({'error': 'Not enough stock data for this ticker.', 'headlines': headlines}), 400

    X, y = create_sequences(data_df['Sentiment_Score'].values)
    if len(X) == 0:
        headlines = fetch_ticker_news(ticker)
        return jsonify({'error': 'Not enough data for sequence creation.', 'headlines': headlines}), 400

    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_and_train_model(X, y)

    prediction = model.predict(X[-1].reshape(1, X.shape[1], 1))[0][0]
    signal = "Buy" if prediction > 0 else "Sell"

    headlines = fetch_ticker_news(ticker)  # fetch again for response

    return jsonify({
        'ticker': ticker,
        'prediction': float(prediction),
        'signal': signal,
        'headlines': headlines
    })

if __name__ == '__main__':
    app.run(debug=True)
