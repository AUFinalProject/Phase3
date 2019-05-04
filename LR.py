# Authors Alexey Titov and Shir Bentabou
# Version 1.0
# Date 05.2019

# libraries
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.metrics import accuracy_score, f1_score
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import re
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
import gensim
from sklearn.model_selection import train_test_split
from sklearn import utils
from gensim.models import Doc2Vec
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
from imutils import paths
import csv
import numpy as np
import pytesseract
import argparse
import imutils
import cv2
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

# dictionary for translate PDF language to tessaract language
lan_lst = {
    "en-us": "eng",	"en": "eng",	"en-za": "eng",	"en-gb": "eng",	"en-in": "eng",
    "es-co": "spa",	"es": "spa",	"de-de": "deu",	"fr-fr": "fra",	"fr-ca": "fra"
}

# dictionary for /Root/Lang 1 - except; 2 - a file have not /Root/Lang; 3 - /Root/Lang = ''; 4 - language
ans_list = dict()

# this function update ans_list


def add_ans_list(save_dir, base_filename, filename):
    try:
        name = os.path.join(save_dir, base_filename)
        pdfFile = PdfFileReader(file(filename, 'rb'))
        catalog = pdfFile.trailer['/Root'].getObject()
        if catalog.has_key("/Lang"):
            lang = catalog['/Lang'].getObject()
            if (lang == ''):
                ans_list.update({name: [3, 'None']})
            else:
                lang = lang.lower()
                language = lan_lst.get(lang)
                ans_list.update({name: [4, language]})
        else:
            ans_list.update({name: [2, 'None']})
    except:
        ans_list.update({name: [1, 'None']})

# this function read information from image


def extract_text_image(imgPath):
    # Define config parameter
    # '--oem 1' for using LSTM OCR Engine
    config = ('--oem 1 --psm 3')

    # Read image from disk
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

    # Read /Root/Lang
    values = ans_list.get(imgPath)
    try:
        if (values[0] == 4):
            langs = value[1]
            imagetext = pytesseract.image_to_string(
                img, lang=langs, config=config)
        else:
            imagetext = pytesseract.image_to_string(img, config=config)
        return imagetext
    except Exception as ex:
        print(imgPath)
        print(ex)
        imagetext = "except"
        return imagetext

# this function convert pdf file to jpg file


def convert(dirpdf):
    # dir of folder and filter for pdf files
    files = [f for f in os.listdir(
        dirpdf) if os.path.isfile(os.path.join(dirpdf, f))]
    files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

    # variables for print information
    cnt_files = len(files)
    i = 0
    for filepdf in files:
        try:
            filename = os.path.join(dirpdf, filepdf)
            with tempfile.TemporaryDirectory() as path:
                images_from_path = convert_from_path(
                    filename, output_folder=path, last_page=1, first_page=0)

            base_filename = os.path.splitext(
                os.path.basename(filename))[0] + '.jpg'
            save_dir = 'IMAGES'

            # save image
            for page in images_from_path:
                page.save(os.path.join(save_dir, base_filename), 'JPEG')
            i += 1

            # update ans_list
            add_ans_list(save_dir, base_filename, filename)

            # show an update every 50 images
            if (i > 0 and i % 50 == 0):
                print("[INFO] processed {}/{}".format(i, cnt_files))
        except Exception:
            # always keep track the error until the code has been clean
            print("[!] Convert PDF to JPEG")
            return False
    return True

# this function write to csv text from pdf


def writeCSV(dirpdf):
    nameCSVfile = 'pdfFiles.csv'
    codec = 'utf-8'
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    save_dir = 'IMAGES'
    with open(nameCSVfile, 'w') as csvFile:

        fields = ['File', 'Kind', 'Text']
        # dir of folder and filter for pdf files
        files = [f for f in os.listdir(
            dirpdf) if os.path.isfile(os.path.join(dirpdf, f))]
        files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

        # variables for print information
        cnt_files = len(files)
        i = 0
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
        for filepdf in files:
            row = dict()
            try:
                filename = os.path.join(dirpdf, filepdf)
                fp = open(filename, 'rb')
                rsrcmgr = PDFResourceManager()
                retstr = StringIO()
                laparams = LAParams()
                device = TextConverter(
                    rsrcmgr, retstr, codec=codec, laparams=laparams)
                # Create a PDF interpreter object.
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                # Process each page contained in the document.
                for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=False):
                    try:
                        interpreter.process_page(page)
                        data = retstr.getvalue()
                        if (len(data) < 2 or len(data) > 100000):
                            base_filename = os.path.splitext(
                                os.path.basename(filename))[0] + '.jpg'
                            imgPath = os.path.join(save_dir, base_filename)
                            data = extract_text_image(imgPath)
                        row = [{'File': filepdf, 'Kind': filepdf.split('.')[
                            0], 'Text':data}]
                    except Exception as ex:
                        print(filepdf)
                        print(ex)
                        base_filename = os.path.splitext(
                            os.path.basename(filename))[0] + '.jpg'
                        imgPath = os.path.join(save_dir, base_filename)
                        data = extract_text_image(imgPath)
                        row = [{'File': filepdf, 'Kind': filepdf.split('.')[
                            0], 'Text':data}]
                    break
                # Cleanup
                device.close()
                retstr.close()
            except Exception as ex:
                print(filepdf)
                print(ex)
                row = [{'File': filepdf, 'Kind': filepdf.split(
                    '.')[0], 'Text':'Exception'}]
            i += 1
            # show an update every 50 pdf
            if (i > 0 and i % 50 == 0):
                print("[INFO] processed {}/{}".format(i, cnt_files))
            writer.writerows(row)
    csvFile.close()

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
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    args = vars(ap.parse_args())
    # define the name of the directory to be created
    path = "IMAGES"
    try:
        os.mkdir(path)
    except OSError:
        print(
            "[!] Creation of the directory %s failed, maybe the folder is exist" % path)
    else:
        print("[*] Successfully created the directory %s " % path)
    arg = os.path.join(os.getcwd(), args["dataset"])
    result = convert(arg)
    if (result):
        print("[*] Succces convert pdf files")
    else:
        print("[!] Whoops. something wrong dude. enable err var to track it")
        sys.exit()
    print("[*] Start write to CSV file")
    writeCSV(arg)
    print("[*] Writing completed")

    df = pd.read_csv('pdfFiles.csv')
    df = df[['Text', 'Kind']]
    df = df[pd.notnull(df['Text'])]
    df.rename(columns={'Text': 'Text'}, inplace=True)
    df.head(10)

    df.shape
    df.index = range(502)
    df['Kind'].apply(lambda x: len(x.split(' '))).sum()
    df['Text'] = df['Text'].apply(cleanText)
    df['Text'][20]

    train, test = train_test_split(df, test_size=0.25, random_state=42)

    train_tagged = train.apply(lambda r: TaggedDocument(
        words=tokenize_text(r['Text']), tags=[r.Kind]), axis=1)
    test_tagged = test.apply(lambda r: TaggedDocument(
        words=tokenize_text(r['Text']), tags=[r.Kind]), axis=1)
    train_tagged.values[30]

    # DBOW
    print("DBOW")
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5,
                         hs=0, min_count=2, sample=0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(
            train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    y_train, X_train = vec_for_learning(model_dbow, train_tagged)
    y_test, X_test = vec_for_learning(model_dbow, test_tagged)
    logreg = LogisticRegression(
        solver='lbfgs', multi_class='auto', max_iter=1000, n_jobs=1, C=1e5)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('  Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('  Testing F1 score: {}'.format(
        f1_score(y_test, y_pred, average='weighted')))

    # Distributed Memory with Averaging
    print("Distributed Memory with Averaging")
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10,
                        negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dmm.train(utils.shuffle([x for x in tqdm(
            train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha

    y_train, X_train = vec_for_learning(model_dmm, train_tagged)
    y_test, X_test = vec_for_learning(model_dmm, test_tagged)

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    print('  Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('  Testing F1 score: {}'.format(
        f1_score(y_test, y_pred, average='weighted')))

    # Union
    print("Union")
    model_dbow.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(
        keep_doctags_vectors=True, keep_inference=True)

    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

    y_train, X_train = get_vectors(new_model, train_tagged)
    y_test, X_test = get_vectors(new_model, test_tagged)

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    print('  Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('  Testing F1 score: {}'.format(
        f1_score(y_test, y_pred, average='weighted')))
