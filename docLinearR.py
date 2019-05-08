# Authors Alexey Titov and Shir Bentabou
# Version 1.0
# Date 05.2019

# libraries
import gensim
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.pipeline import Pipeline
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import csv
import numpy as np
import os
import tempfile
import sys
from importlib import reload
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# Do this in a separate python interpreter session, since you only have to do it once
# nltk.download('punkt')
# Do this in your ipython notebook or analysis script


cores = multiprocessing.cpu_count()
# fix UnicodeEncodeError
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")


# get vector
def get_vectors(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(
        *[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

# create vector
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(
        *[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

# tokenizer for text
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if (len(word) < 2):
                continue
            tokens.append(word.lower())
    return tokens

# this function clean text from no wanted symbols
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


if __name__ == "__main__":

    print("[*] Start read from CSV file")
    df = pd.read_csv('pdfFiles_linear.csv')
    df = df[['Text', 'Kind']]
    df = df[pd.notnull(df['Text'])]
    df.rename(columns={'Text': 'Text'}, inplace=True)

    df.shape
    df.index = range(502)
    df['Text'] = df['Text'].apply(cleanText)

    train, test = train_test_split(df, test_size=0.25, random_state=42)

    train_tagged = train.apply(lambda r: TaggedDocument(
        words=tokenize_text(r['Text']), tags=[r.Kind]), axis=1)
    test_tagged = test.apply(lambda r: TaggedDocument(
        words=tokenize_text(r['Text']), tags=[r.Kind]), axis=1)
    train_tagged.values[30]
    print("[*] Stop read from CSV file")
    # DBOW
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5,
                         hs=0, min_count=2, sample=0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(
            train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    # Distributed Memory with Averaging
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10,
                        negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dmm.train(utils.shuffle([x for x in tqdm(
            train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha


    # Union
    print("Union")
    model_dbow.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)

    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

    y_train, X_train = get_vectors(new_model, train_tagged)
    y_test, X_test = get_vectors(new_model, test_tagged)

    # Linear Regression

    # create linear regression object
    linreg = Pipeline([('clf', linear_model.LinearRegression()),])
  
    # train the model using the training sets 
    linreg.fit(X_train, y_train)
  
    # variance score: 1 means perfect prediction 
    print('Variance score: {}'.format(linreg.score(X_test, y_test)))
	
    # Make predictions using the testing set
    diabetes_y_pred = linreg.predict(X_test)
    # Explained variance score: 1 is perfect prediction 
    print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))
