from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

def fetch_ticker_news_ordered(ticker, total_headlines):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    driver.get(url)

    headlines = []
    seen = set()
    scroll_pause_time = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while len(headlines) < total_headlines:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        
        elems = driver.find_elements(By.CSS_SELECTOR, 'h3.clamp')
        for elem in elems:
            text = elem.text.strip()
            if text and text not in seen:
                headlines.append(text)
                seen.add(text)
            if len(headlines) >= total_headlines:
                break
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.quit()
    return headlines[:total_headlines]


if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper()
    days = 15
    headlines_per_day = 12
    total_headlines = days * headlines_per_day

    print(f"Fetching {total_headlines} headlines for ticker '{ticker}'...")

    headlines = fetch_ticker_news(ticker, total_headlines)
    print(f"\nNumber of headlines fetched: {len(headlines)}\n")

    print("First 30 headlines (or fewer if less available):\n")
    for i, headline in enumerate(headlines, start=1):
        print(f"{i}. {headline}")
