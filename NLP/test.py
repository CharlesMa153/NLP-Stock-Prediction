from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

def fetch_ticker_news_selenium(ticker, total_headlines=30):
    url = f'https://finance.yahoo.com/quote/{ticker}/news'

    options = Options()
    options.headless = True  # Run Chrome in headless mode (no GUI)

    # Make sure you have chromedriver installed and in your PATH
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Wait for page to load JS content (adjust sleep time if needed)
    time.sleep(5)

    # Grab headline elements by CSS selector
    elements = driver.find_elements(By.CSS_SELECTOR, 'h3')

    print(f"Found {len(elements)} headline elements (raw)")

    # Extract text, limit to total_headlines
    headlines = [el.text for el in elements if el.text.strip()][:total_headlines]

    print(f"Returning {len(headlines)} headlines:")
    for i, h in enumerate(headlines[:5]):
        print(f"{i+1}. {h}")

    driver.quit()
    return headlines if headlines else ["No news found"]

if __name__ == '__main__':
    ticker = 'AAPL'  # Example ticker
    fetch_ticker_news_selenium(ticker)
