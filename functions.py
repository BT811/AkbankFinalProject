import yfinance as yf
import matplotlib.pyplot as plt
from HisseTahmin import train_and_predict_next_n_days

ticker = None

def hello():
    return "Merhaba!"

def goodby():
    return "Hoşçakal!"

def enter_ticker():
    global ticker
    ticker = input("Hisse senedi kodunu giriniz:")
    if yf.Ticker(ticker).history(period ="1y").empty:
        print(f"{ticker} sembolü için veri bulunamadı. Tekrar deneyin")
        enter_ticker()   

def graph():
    global ticker
    if ticker == None:
        enter_ticker()
        
    data = yf.Ticker(ticker).history(period="1y")
    data = yf.Ticker(ticker).history(period="1y")
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data.Close)
    plt.title(f"{ticker} Hisse Senedi Fiyatı (1 Yıllık)")
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat(Dolar)")
    plt.grid(True)
    
    return plt.gcf()

def calculate_RSI():
    global ticker
    if ticker == None:
        enter_ticker()
    data = yf.Ticker(ticker).history(period="1y")
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
    return str(rsi)


def calculate_EMA():
    global ticker
    if ticker == None:
        enter_ticker()
    window = int(input("Kaç günlük Üssel Hareketli Ortalamayı (EMA) görmek istiyorsunuz? "))
    data = yf.Ticker(ticker).history(period="1y")
    ema = data['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
    return str(ema)

def calculate_SMA():
    global ticker
    if ticker == None:
        enter_ticker()
    window = int(input("Son kaç günün kapanış fiyat ortalamasını (SMA) istiyorsunuz?"))
    data = yf.Ticker(ticker).history(period="1y")
    sma = data['Close'].rolling(window=window).mean().iloc[-1]

    return str(sma)

def calculate_MACD():
    global ticker
    if ticker == None:
        enter_ticker()
    data = yf.Ticker(ticker).history(period="1y")['Close']
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()

    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal

    result =  f"{MACD[-1]},{signal[-1],{MACD_histogram[-1]}}"
    return result

def model_tahmin():
    if ticker is None:
        enter_ticker()
    days = int(input("Kaç günlük açılışı değerini modelinize hesaplatmak istiyorsunuz?"))
    result_text =  ""
    predicted_open_prices = train_and_predict_next_n_days(ticker,days)
    for i, price in enumerate(predicted_open_prices):
        result_text += f"{i+1}. günün tahmin edilen açılış fiyatı: {price}\n"

    return result_text