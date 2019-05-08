# Authors Alexey Titov and Shir Bentabou
# Version 1.0
# Date 05.2019

# libraries
import csv
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
# Do this in a separate python interpreter session, since you only have to do it once
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



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



# Kind: 0 - malware, 1- white
df = pd.read_csv('pdfFiles_linear.csv')
df = df[['Text', 'Kind']]
df = df[pd.notnull(df['Text'])]

plt.figure(figsize=(10,4))
df.Kind.value_counts().plot(kind='bar');

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))    
df['Text'] = df['Text'].apply(clean_text)

df['Text'].apply(lambda x: len(x.split(' '))).sum()
X = df.Text
y = df.Kind
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)


# Linear Regression
print("Linear Regression")
linreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', linear_model.LinearRegression()),
               ])
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(linreg.score(X_test, y_test))) 
