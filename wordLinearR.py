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
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from itertools import islice
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

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


df = pd.read_csv('pdfFiles_linear.csv')
df = df[['Text', 'Kind']]
df = df[pd.notnull(df['Text'])]
df.rename(columns={'Text': 'Text'}, inplace=True)


my_tags = ['white','mal']
plt.figure(figsize=(10,4))
df.Kind.value_counts().plot(kind='bar');

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))    
df.shape
df.index = range(502)
df['Text'] = df['Text'].apply(clean_text)

# Word2vec and Linear Regression
# https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit GoogleNews-vectors-negative300.bin.gz
# 3 million words * 300 features * 4bytes/feature = ~3.35GB
wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)
wv.init_sims(replace=True)
list(islice(wv.vocab, 13030, 13050))
train, test = train_test_split(df, test_size=0.25, random_state = 42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['Text']), axis=1).values

X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)

# create linear regression object 
linreg = linear_model.LinearRegression()
linreg = linreg.fit(X_train_word_average, train['Kind'])
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(linreg.score(X_test_word_average, test.Kind)))

# Make predictions using the testing set
diabetes_y_pred = linreg.predict(X_test_word_average)
# Explained variance score: 1 is perfect prediction 
print('Variance score: %.2f' % r2_score(test.Kind, diabetes_y_pred))
