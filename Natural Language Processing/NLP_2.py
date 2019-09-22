# -*- coding: utf-8 -*-

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
def word_features(word):
    return {'items': word}
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    if(dataset['Liked'][i]==0):
        result = "negative";
    else:
        result = "positive"
    for word in review:
        corpus.append((word_features(word),result));

import random
random.shuffle(corpus)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(corpus, test_size = 0.20, random_state = 0)

from nltk.classify import MaxentClassifier
classifier = MaxentClassifier.train(X_train)

# Predicting the Test set results
y_pred = classifier.classify(word_features("first"))

# Making the Confusion Matrix
print(nltk.classify.accuracy(classifier, X_test))
