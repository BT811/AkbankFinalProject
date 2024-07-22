import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from functions import (
    hello, goodby, enter_ticker, graph, calculate_RSI, 
    calculate_EMA, calculate_SMA, calculate_MACD,
    model_tahmin,res_sup_graph,myStocks,
    user_opertation_graph
)
import matplotlib.pyplot as plt


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

def get_responde(intents_list, intent_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intent_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = i["response"][0]
            break
    return result

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

print("Başla")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_responde(ints, intents)
    if res in mappings:
        answer = mappings[res]()

        if isinstance(answer, plt.Figure):  # Eğer answer bir plt.Figure objesi ise
            plt.show()  # Grafik gösterilir
            plt.close()
        elif answer is not None:
            print(answer)  # Diğer durumlarda answer'ı bas(String durumları)

