# Authors Alexey Titov and Shir Bentabou
# Version 2.0
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
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
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


df = pd.read_csv('pdfFiles_01.csv')
df = df[['Text', 'Kind']]
df = df[pd.notnull(df['Text'])]

my_tags = ['0','1']
plt.figure(figsize=(10,4))
df.Kind.value_counts().plot(kind='bar')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))    
df['Text'] = df['Text'].apply(clean_text)

df['Text'].apply(lambda x: len(x.split(' '))).sum()
X = df.Text
y = df.Kind
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

# Naive Bayes Classifier for Multinomial Models
print("Naive Bayes Classifier for Multinomial Models")
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
cm = confusion_matrix(y_test, y_pred)
# the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)

# Linear Support Vector Machine
print("Linear Support Vector Machine")
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=200, tol=1e-3)),
               ])
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
cm = confusion_matrix(y_test, y_pred)
# the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)

# Logistic Regression
print("Logistic Regression")
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
cm = confusion_matrix(y_test, y_pred)
# the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)

# Random Forest
print("Random Forest")
ranfor = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier(n_estimators=30, random_state=0)),
               ])
ranfor.fit(X_train, y_train)
y_pred = ranfor.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
cm = confusion_matrix(y_test, y_pred)
# the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)

# K-Nearest Neighbors
print("K-Nearest Neighbors")
knn = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', KNeighborsClassifier(n_neighbors=3)),
               ])
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
cm = confusion_matrix(y_test, y_pred)
# the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)

# Multi-layer Perceptron
print("Multi-layer Perceptron")
mlp = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1, tol=0.000000001)),
               ])
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
cm = confusion_matrix(y_test, y_pred)
# the count of true negatives is A00, false negatives is A10, true positives is A11 and false positives is A01
print('confusion matrix:\n %s' % cm)

