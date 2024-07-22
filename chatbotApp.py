import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
from tkinter import simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from HisseTahmin import train_and_predict_next_n_days
import mplfinance as mpf
from graphFuntions import res_sup_line_graph, show_user_opertation_graph
from myStocksFunctions import stock_information
import pandas as pd

graph_canvas = None
ticker = None
data = None
"""MODEL KISMI"""

intents = json.loads(open('intentsEXMP.json', "r", encoding="utf-8").read())

words = pickle.load(open("wordsExp.pkl", "rb"))
classes = pickle.load(open("classesExp.pkl", "rb"))

model = tf.keras.models.load_model('chatbot02.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intent_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intent_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = i["response"][0]
            break
    return result
"""FONKSİYONLAR"""
def hello():
    chat_window.insert(tk.END, "Bot: Merhaba! Size şu konularda yardım edebilirim.\n Hisse Grafiği \n EMA Üssel Hareketli Ortalama\n RSI Göreceli Güç Endeks\n MACD Moving Average Convergence Divergence \n SMA Basit Hareketli Ortalama \n Hisse Açılış Değer Tahmini. \n Destek-Direnç çizgileri. \n Portföy bilgisi. \n Alım-satım yaptığınız noktalar. \n")

def goodby():
    chat_window.insert(tk.END, "Bot: Hoşçakal!\n")

def enter_ticker():
    global ticker
    global graph_canvas
    global data

    ticker = simpledialog.askstring("Input", "Hisse senedi kodunu giriniz:")
    data = yf.Ticker(ticker).history(period ="1y")

    if not ticker or data.empty:
        okcancek = messagebox.askokcancel("Error", f"{ticker} sembolü için veri bulunamadı. Tekrar deneyebilirsiniz.")
        if okcancek:
            enter_ticker()
        else:
            ticker = None
    else:
        chat_window.insert(tk.END, f"{ticker}  için işlemlerinize devam edebilirsiniz.\n")

        if graph_canvas:
            graph_canvas.get_tk_widget().pack_forget()
            plt.close()
            graph_canvas = None

    
    
def remove_graph():
    global graph_canvas
    if graph_canvas:
        graph_canvas.get_tk_widget().pack_forget()
        plt.close()
        graph_canvas = None  
          



def graph():
    global ticker
    global graph_canvas
    global data

    if ticker is None:
        enter_ticker()
    if ticker:

        remove_graph()

        plt.style.use("dark_background")

        colors = mpf.make_marketcolors(up="#00ff00", down="#ff0000", wick="inherit", edge="inherit", volume="in")
        mpf_style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=colors)

        fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, returnfig=True)

        graph_canvas = FigureCanvasTkAgg(fig, master=root)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack()
        plt.close()

def calculate_RSI():
    global ticker
    if ticker is None:
        enter_ticker()
    if ticker:
        data = yf.Ticker(ticker).history(period="1y")
        delta = data['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=14-1, adjust=False).mean()
        ema_down = down.ewm(com=14-1, adjust=False).mean()
        rs = ema_up / ema_down

        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        chat_window.insert(tk.END, "Bot: " + f"RSI: {rsi}""\n", "bot")

def calculate_EMA():
    global ticker
    if ticker is None:
        enter_ticker()
    if ticker:
        window = simpledialog.askinteger("Input", "Kaç günlük Üssel Hareketli Ortalamayı (EMA) görmek istiyorsunuz?")
        data = yf.Ticker(ticker).history(period="1y")
        ema = data['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
        chat_window.insert(tk.END, "Bot: " +  f"{window} günlük EMA: {ema}"+ "\n", "bot")


def calculate_SMA():
    global ticker
    if ticker is None:
        enter_ticker()
    if ticker:
        window = simpledialog.askinteger("Input", "Son kaç günün kapanış fiyat ortalamasını (SMA) istiyorsunuz?")
        data = yf.Ticker(ticker).history(period="1y")
        sma = data['Close'].rolling(window=window).mean().iloc[-1]
        chat_window.insert(tk.END, f"{window} günlük SMA: {sma}\n")
        chat_window.insert(tk.END, "Bot: " + f"{window} günlük SMA: {sma}"+ "\n", "bot")

def calculate_MACD():
    global ticker
    if ticker is None:
        enter_ticker()
    if ticker:
        data = yf.Ticker(ticker).history(period="1y")['Close']
        short_EMA = data.ewm(span=12, adjust=False).mean()
        long_EMA = data.ewm(span=26, adjust=False).mean()

        MACD = short_EMA - long_EMA
        signal = MACD.ewm(span=9, adjust=False).mean()
        MACD_histogram = MACD - signal

        chat_window.insert(tk.END, "Bot: " f"MACD: {MACD.iloc[-1]}, Signal: {signal.iloc[-1]}, Histogram: {MACD_histogram.iloc[-1]}\n", "bot")

def model_tahmin():
    if ticker is None:
        enter_ticker()
    days = simpledialog.askinteger("Input", "Kaç günlük açılışı değerini modelinize hesaplatmak istiyorsunuz?")
    result_text =  ""
    predicted_open_prices = train_and_predict_next_n_days(ticker,days)
    for i, price in enumerate(predicted_open_prices):
        result_text += f"{i+1}. günün tahmin edilen açılış fiyatı: {price}\n"

    chat_window.insert(tk.END, "Bot: " + result_text + "\n", "bot")


def res_sup_graph():
    global ticker
    global data
    global graph_canvas

    if ticker is None:
        enter_ticker()

    remove_graph()

    fig = res_sup_line_graph(data)
    graph_canvas = FigureCanvasTkAgg(fig, master=root)
    graph_canvas.draw()
    graph_canvas.get_tk_widget().pack()
    plt.close()

def myStocks():
    user_operations = pd.read_csv("kullanıcı_işlemleri.csv")
    chat_window.insert(tk.END, str(stock_information(user_operations))+"\n","bot")

def user_opertation_graph():
    global ticker
    global data
    global graph_canvas

    if ticker is None:
        enter_ticker()

    remove_graph()

    user_operations = pd.read_csv("kullanıcı_işlemleri.csv", parse_dates=['İşlem Tarihi'], index_col='İşlem Tarihi')
    fig = show_user_opertation_graph(ticker, data, user_operations)
    if isinstance(fig, str):
        chat_window.insert(tk.END, fig, "bot")
    else:
        graph_canvas = FigureCanvasTkAgg(fig, master=root)
        graph_canvas.draw()
        graph_canvas.get_tk_widget().pack()
        plt.close()
    


mappings = {
    "merhaba": hello,
    "hoşçakal": goodby,
    "grafik": graph,
    "rsı": calculate_RSI,
    "ema": calculate_EMA,
    "sma": calculate_SMA,
    "macd": calculate_MACD,
    "ticker": enter_ticker,
    "tahmin": model_tahmin,
    "destekdireç": res_sup_graph,
    "hissem" : myStocks,
    "kullanıcıişlemleri":user_opertation_graph
}

def send_message(event=None):
    message = entry.get()
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "Kullanıcı: " + message + "\n", "user")
    entry.delete(0, tk.END)
    ints = predict_class(message)
    res = get_response(ints, intents)
    mappings[res]()
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)

# Tkinter GUI oluşturma
root = tk.Tk()
root.title("Chat Bot")


# Chat penceresi
chat_window = scrolledtext.ScrolledText(root, state=tk.DISABLED, width=80, height=20, wrap=tk.WORD, font=("Helvetica", 12), bg="#333333", fg="#CCCCCC", padx=10, pady=10)
chat_window.tag_config('user', foreground='#FFFFFF')  # Kullanıcı mesajları için beyaz renk
chat_window.tag_config('bot', foreground='#FFFFFF')  # Bot mesajları için
chat_window.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Kullanıcı giriş kutusu
entry_frame = tk.Frame(root, bg='#2E2E2E')
entry_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
entry = tk.Entry(entry_frame, width=60, font=("Helvetica", 12), bg="#555555", fg="#CCCCCC", relief=tk.FLAT, bd=2, insertbackground="#CCCCCC")
entry.pack(side=tk.LEFT, padx=5, pady=5, ipady=5, expand=True, fill=tk.X)
entry.bind("<Return>", send_message)

# Gönder düğmesi
send_button = tk.Button(entry_frame, text="Gönder", font=('Helvetica', 12), bg="#4CAF50", fg="white", relief=tk.FLAT, activebackground="#45a049", padx=10, pady=5, command=send_message)
send_button.pack(side=tk.LEFT, padx=5, pady=5)

# Arka plan rengi
root.configure(background='#2E2E2E')

# Tam ekran yapıldığında uyumsuzluğu engellemek için minimum boyut ayarı
root.minsize(800, 600)

# Tkinter ana döngüsü
root.mainloop()