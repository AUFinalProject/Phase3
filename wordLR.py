# Authors Alexey Titov and Shir Bentabou
# Version 1.0
# Date 05.2019

# libraries
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
# Do this in a separate python interpreter session, since you only have to do it once
# nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from itertools import islice
# NB for negative
from sklearn.naive_bayes import GaussianNB
# NB only positive
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub('', text)
    # delete stopwors from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(
        np.array(mean).mean(
            axis=0)).astype(
        np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


df = pd.read_csv('pdfFiles_01.csv')
df = df[['Text', 'Kind']]
df = df[pd.notnull(df['Text'])]

my_tags = ['0', '1']
plt.figure(figsize=(10, 4))
df.Kind.value_counts().plot(kind='bar')

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
df.shape
df.index = range(502)
#df['Kind'].apply(lambda x: len(x.split(' '))).sum()
df['Text'] = df['Text'].apply(clean_text)


X = df.Text
y = df.Kind
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# Word2vec
# https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit GoogleNews-vectors-negative300.bin.gz
# 3 million words * 300 features * 4bytes/feature = ~3.35GB
wv = gensim.models.KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)
wv.init_sims(replace=True)
list(islice(wv.vocab, 13030, 13050))
train, test = train_test_split(df, test_size=0.25, random_state=42)

test_tokenized = test.apply(
    lambda r: w2v_tokenize_text(
        r['Text']), axis=1).values
train_tokenized = train.apply(
    lambda r: w2v_tokenize_text(
        r['Text']), axis=1).values

X_train_word_average = word_averaging_list(wv, train_tokenized)
X_test_word_average = word_averaging_list(wv, test_tokenized)


# Logistic Regression
print("Logistic Regression")
logreg = Pipeline([('clf', LogisticRegression(
    solver='lbfgs', multi_class='auto', max_iter=1000, n_jobs=1, C=1e5)), ])
logreg.fit(X_train_word_average, train['Kind'])
y_pred = logreg.predict(X_test_word_average)

print('accuracy %s' % accuracy_score(y_pred, test.Kind))
print(classification_report(test.Kind, y_pred, target_names=my_tags))
cm = confusion_matrix(test.Kind, y_pred)
# the count of true negatives is A00, false negatives is A10, true
# positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)
print("\n\n")

# Naive Bayes Classifier for Gaussian Model
print("Naive Bayes Classifier for Gaussian Model")
nb = Pipeline([('clf', GaussianNB()),
               ])
nb.fit(X_train_word_average, train['Kind'])
y_pred = nb.predict(X_test_word_average)

print('accuracy %s' % accuracy_score(y_pred, test.Kind))
print(classification_report(test.Kind, y_pred, target_names=my_tags))
cm = confusion_matrix(test.Kind, y_pred)
# the count of true negatives is A00, false negatives is A10, true
# positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)
print("\n\n")

# Linear Support Vector Machine
print("Linear Support Vector Machine")
sgd = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2',
                                      alpha=1e-3, random_state=42, max_iter=200, tol=1e-3)), ])
sgd.fit(X_train_word_average, train['Kind'])
y_pred = sgd.predict(X_test_word_average)

print('accuracy %s' % accuracy_score(y_pred, test.Kind))
print(classification_report(test.Kind, y_pred, target_names=my_tags))
cm = confusion_matrix(test.Kind, y_pred)
# the count of true negatives is A00, false negatives is A10, true
# positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)
print("\n\n")

# Random Forest
print("Random Forest")
ranfor = Pipeline([
    ('clf', RandomForestClassifier(n_estimators=30, random_state=0)),
])
ranfor.fit(X_train_word_average, train['Kind'])
y_pred = ranfor.predict(X_test_word_average)

print('accuracy %s' % accuracy_score(y_pred, test.Kind))
print(classification_report(test.Kind, y_pred, target_names=my_tags))
cm = confusion_matrix(test.Kind, y_pred)
# the count of true negatives is A00, false negatives is A10, true
# positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)
print("\n\n")

# K-Nearest Neighbors
print("K-Nearest Neighbors")
knn = Pipeline([
    ('clf', KNeighborsClassifier(n_neighbors=3)),
])
knn.fit(X_train_word_average, train['Kind'])
y_pred = knn.predict(X_test_word_average)

print('accuracy %s' % accuracy_score(y_pred, test.Kind))
print(classification_report(test.Kind, y_pred, target_names=my_tags))
cm = confusion_matrix(test.Kind, y_pred)
# the count of true negatives is A00, false negatives is A10, true
# positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)
print("\n\n")

# Multi-layer Perceptron
print("Multi-layer Perceptron")
mlp = Pipeline([('clf',
                 MLPClassifier(activation='relu',
                               solver='lbfgs',
                               alpha=1e-5,
                               hidden_layer_sizes=(15,
                                                   ),
                               random_state=1,
                               tol=0.000000001)),
                ])
mlp.fit(X_train_word_average, train['Kind'])
y_pred = mlp.predict(X_test_word_average)

print('accuracy %s' % accuracy_score(y_pred, test.Kind))
print(classification_report(test.Kind, y_pred, target_names=my_tags))
cm = confusion_matrix(test.Kind, y_pred)
# the count of true negatives is A00, false negatives is A10, true
# positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)
print("\n\n")
