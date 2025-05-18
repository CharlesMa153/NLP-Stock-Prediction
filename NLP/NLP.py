import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date, timedelta

from flask import Flask, render_template, jsonify, request

# Download VADER lexicon once
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

app = Flask(__name__, template_folder='templates')


def fetch_ticker_news(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Select h3 tags with class 'clamp  yf-1y7058a' (two spaces)
    headlines = [h3.get_text(strip=True) for h3 in soup.select('h3.clamp.yf-1y7058a')]
    
    return headlines if headlines else ["No news found"]



def get_sentiment_score(headlines):
    scores = [sia.polarity_scores(h)['compound'] for h in headlines]
    return scores


def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    close_col = next((col for col in df.columns if 'Close' in col), None)
    if not close_col:
        raise ValueError("Close column not found in stock data.")
    df = df[[close_col]].rename(columns={close_col: 'Close'}).reset_index()
    return df


def prepare_dataset(ticker):
    today = date.today()
    start_date = today - timedelta(days=9)  # last 10 days including today
    end_date = today + timedelta(days=1)  # up to tomorrow to get today's close if available

    print(f"Preparing data for {ticker} from {start_date} to {today}")

    stock_df = fetch_stock_data(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if stock_df.empty:
        print("No stock data found.")
        return None

    sentiment_rows = []
    for i in range(10):
        target_date = today - timedelta(days=9 - i)
        headlines = fetch_ticker_news(ticker)
        scores = get_sentiment_score(headlines)
        avg_score = np.mean(scores) if scores else 0
        sentiment_rows.append({'Date': target_date, 'Sentiment_Score': avg_score})

    sentiment_df = pd.DataFrame(sentiment_rows)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    merged = pd.merge(stock_df[['Date', 'Close']], sentiment_df, on='Date', how='left')
    merged['Sentiment_Score'].fillna(0, inplace=True)
    return merged


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

    # DELETE LATER
    headlines = fetch_ticker_news(ticker)

    data_df = prepare_dataset(ticker)
    if data_df is None or len(data_df) < 2:
        return jsonify({'error': 'Not enough data for this ticker.', 'headlines': headlines}), 400

    X, y = create_sequences(data_df['Sentiment_Score'].values)
    if len(X) == 0:
        return jsonify({'error': 'Not enough data for sequence creation.', 'headlines': headlines}), 400

    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_and_train_model(X, y)

    prediction = model.predict(X[-1].reshape(1, X.shape[1], 1))[0][0]
    signal = "Buy" if prediction > 0 else "Sell"

    return jsonify({
        'ticker': ticker,
        'prediction': float(prediction),
        'signal': signal,
        'headlines': headlines
    })


if __name__ == '__main__':
    app.run(debug=True)
