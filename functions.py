import yfinance as yf
import matplotlib.pyplot as plt
from HisseTahmin import train_and_predict_next_n_days
import mplfinance as mpf
from graphFuntions import res_sup_line_graph, stock_graph, show_user_opertation_graph
from myStocksFunctions import stock_information
import pandas as pd

global ticker
ticker=None
global fig 
fig=None
global data 
data=None


def hello():
    return "Merhaba!"

def goodby():
    return "Hoşçakal!"

def enter_ticker():
    global ticker
    global fig 
    global data
    fig = None
    ticker = input("Hisse senedi kodunu giriniz:")
    data = yf.Ticker(ticker).history(period ="1y")
    if data.empty:
        print(f"{ticker} sembolü için veri bulunamadı. Tekrar deneyin")
        enter_ticker() 
    
    
     

def graph():
    global ticker  # Ticker sembolü büyük harflerle yazılmalı
    global data
    if ticker == None:
        enter_ticker()

    return stock_graph(data)

def calculate_RSI():
    global ticker
    global data
    if ticker == None:
        enter_ticker()
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
    global data
    if ticker == None:
        enter_ticker()
    window = int(input("Kaç günlük Üssel Hareketli Ortalamayı (EMA) görmek istiyorsunuz? "))
    ema = data['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
    return str(ema)

def calculate_SMA():
    global ticker
    global data
    if ticker == None:
        enter_ticker()
    window = int(input("Son kaç günün kapanış fiyat ortalamasını (SMA) istiyorsunuz?"))

    sma = data['Close'].rolling(window=window).mean().iloc[-1]

    return str(sma)

def calculate_MACD():
    global ticker
    global data
    if ticker == None:
        enter_ticker()
    df = data['Close']
    short_EMA = df.ewm(span=12, adjust=False).mean()
    long_EMA = df.ewm(span=26, adjust=False).mean()

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

def res_sup_graph():
    global ticker
    global data

    if ticker is None:
        enter_ticker()
    
    
    return res_sup_line_graph(data)
    
def myStocks():
    user_operations = pd.read_csv("kullanıcı_işlemleri.csv")
    return str(stock_information(user_operations))

def user_opertation_graph():
    global ticker
    global data
    if ticker is None:
        enter_ticker()
    user_operations = pd.read_csv("kullanıcı_işlemleri.csv", parse_dates=['İşlem Tarihi'], index_col='İşlem Tarihi')

    return show_user_opertation_graph(ticker, data, user_operations)
