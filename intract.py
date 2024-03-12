import tensorflow as tf
from tensorflow import keras
import nltk
import random
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TweetTokenizer
import json
import numpy as np
import pickle

# nltk.download('punkt')  # Uncomment if punkt is not downloaded
tokens = TweetTokenizer()

from nltk.stem import PorterStemmer
ps = PorterStemmer()

with open("data.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent['patterns']:
        wrds = tokens.tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [ps.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [ps.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

model = keras.Sequential([
    keras.layers.Input(shape=(len(training[0]),)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model.load_weights("main_project_model.h5")
except Exception as e:
    print("Exception:", e)
    history = model.fit(training, output, epochs=500, batch_size=8, verbose=1)
    model.save("main_project_model.h5")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = tokens.tokenize(s)
    s_words = [ps.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chat(inp):
    results = model.predict(np.array([bag_of_words(inp, words)]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responces"]
                output = random.choice(responses)
                return output
    else:
        return "I didn't get that, try again!"
user_input=input("Ask me Anything : ")
print(chat(user_input))
