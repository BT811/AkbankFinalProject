import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

import zeyrek
"""
from snowballstemmer import TurkishStemmer
turkStem=TurkishStemmer()
analayzer = zeyrek.MorphAnalyzer()
"""

intents = json.loads(open('intentsEXMP.json',"r",encoding="utf-8").read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

"""
results =[analayzer.lemmatize(word) for word in words if word not in ignoreLetters]
words = [ result[0][1][0]  for result in results ]
"""

words =[word for word in words if word not in ignoreLetters]


words = sorted(set(words))
classes = sorted(set(classes))


pickle.dump(words, open('wordsExp.pkl', 'wb'))
pickle.dump(classes, open('classesExp.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [word.lower() for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

print(training)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(trainX, trainY, epochs=300, batch_size=5, verbose=1)
model.save('chatbot02.h5')
print('Done')