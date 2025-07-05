# NLP Stock Prediction Web App

This Flask web application predicts stock price movements by combining financial news sentiment analysis with historical stock prices. It scrapes news headlines using Selenium, analyzes sentiment with FinBERT, and uses an LSTM neural network to forecast stock closing prices.

## Features
- News Scraping: Uses Selenium WebDriver in headless mode to scrape up to 500 recent news headlines for a given stock ticker from Yahoo Finance.

- Sentiment Analysis: Applies the FinBERT transformer model to assign daily sentiment scores to the collected news headlines.

- Stock Data: Downloads historical stock closing prices using yFinance.

- Data Preparation: Combines daily sentiment scores with stock prices, normalizes features, and creates sequential datasets.

- LSTM Model: Builds and trains an LSTM neural network to predict next-day closing prices based on past price and sentiment data.

- Prediction Signal: Generates a buy/sell signal based on whether the predicted next day’s price is higher or lower than the latest closing price.

- API Endpoint: A POST /analyze endpoint accepts a JSON payload with a stock ticker and returns JSON containing predictions, sentiment, and price data.

## Technologies Used
- Python

- Flask for the web server and API

- Selenium for web scraping (headless Chrome)

- Transformers library with pre-trained FinBERT model for sentiment analysis

- PyTorch as backend for FinBERT

- yFinance for historical stock price data

- TensorFlow/Keras for LSTM model building and training

- scikit-learn for MinMaxScaler normalization

- Pandas & NumPy for data manipulation
