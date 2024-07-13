import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from HisseTahmin import train_and_predict_next_n_days


ticker = None

"""MODEL KISMI"""

intents = json.loads(open('intentsEXMP.json',"r",encoding="utf-8").read())

words = pickle.load(open("wordsExp.pkl","rb"))
classes = pickle.load(open("classesExp.pkl","rb"))

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
    res= model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key = lambda x: x[1], reverse = True)
    return_list=[]
    for r in result:
        return_list.append({"intent":classes[r[0]], "probability":str(r[1])})
    return return_list

def get_responde (intents_list, intent_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intent_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = i["response"][0]
            break
    return result


"""METHODLAR"""

def hello():
    print("Merhaba!")

def goodby():
    print("Hoşçakal!")

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
    plt.show()
    plt.close()

def calculate_RSI():
    global ticker
    if ticker == None:
        enter_ticker()
    data = yf.Ticker(ticker).history(period="1y")
    delta = data.diff()
    up = delta.clip(lower = 0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down

    print(str(100-(100 / (1+rs)).iloc[-1]))

def calculate_EMA():
    global ticker
    if ticker == None:
        enter_ticker()
    window = int(input("Kaç günlük Üssel Hareketli Ortalamayı (EMA) görmek istiyorsunuz? "))
    data = yf.Ticker(ticker).history(period="1y")
    print(str(data.ewm(span=window, adjust=False).mean().iloc[-1]))

def calculate_SMA():
    global ticker
    if ticker == None:
        enter_ticker()
    window = int(input("Son kaç günün kapanış fiyat ortalamasını (SMA) istiyorsunuz?"))
    data = yf.Ticker(ticker).history(period="1y")
    print(str(data.rolling(window=window).mean().iloc[-1]))

def calculate_MACD():
    global ticker
    if ticker == None:
        enter_ticker()
    data = yf.Ticker(ticker).history(period="1y").Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()

    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal

    print(f"{MACD[-1]},{signal[-1],{MACD_histogram[-1]}}") 

def model_tahmin():
    if ticker is None:
        enter_ticker()
    days = int(input("Kaç günlük açılışı değerini modelinize hesaplatmak istiyorsunuz?"))
    result_text =  ""
    predicted_open_prices = train_and_predict_next_n_days(ticker,days)
    for i, price in enumerate(predicted_open_prices):
        result_text += f"{i+1}. günün tahmin edilen açılış fiyatı: {price}\n"

    print(result_text)

mappings={
    "merhaba": hello,
    "hoşçakal": goodby,
    "grafik": graph,
    "rsı": calculate_RSI,
    "ema": calculate_EMA,
    "sma": calculate_SMA,
    "macd":calculate_MACD,
    "ticker": enter_ticker,
    "tahmin": model_tahmin

}



"""YÜRÜTME"""
print("Başla")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_responde(ints,intents)
    mappings[res]()



