import nltk 
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json

nltk.download("punkt")
stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

print(words)
print(labels)
print(docs)
