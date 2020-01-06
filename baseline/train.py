'''
 --Baseline Model Training Storage
'''
import numpy as np 
import pandas as pd
import sklearn
import os
import json
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
import re
import sys
sys.path.append('../utils')
import prep
from prep import make_tokens, filter_by_count, sent2index, make_vocab, make_word2index
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class Config:
    def __init__(self, folder):
        self.dir = folder

    def load(self, filename):
        path = os.path.join(self.dir, filename)
        with open(path, 'r') as f:
            data = json.load(f)
            f.close()
        return data

    def save(self, filename, data):
        path = os.path.join(self.dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f)
            f.close()


if __name__ == '__main__':
    df = pd.read_csv('../data/clean.csv')
    tag_columns = list(filter(lambda x: re.match(r'^Tag', x), list(df)))

    sentences = list(df['Joint_Text'])
    labels = np.array(df[tag_columns])
    num_classes = np.sum(labels, axis=1)
    y = [1 if num>1 else 0 for num in num_classes]


    tokens = make_tokens(sentences)
    tokens = filter_by_count(tokens, 10000)

    vocab = make_vocab(tokens)
    word2index = make_word2index(vocab)
    index2word = {value:key for key, value in word2index.items()}
    sequences = sent2index(word2index, tokens)

    assert len(sequences) == len(y)

    pairs = list(zip(sequences, y))
    pairs = list(filter(lambda x: len(x[0])>1, pairs))

    sequences = [seq for seq,_ in pairs]
    y = np.array([lab for _,lab in pairs])

    refined_tokens = list(map(lambda x: [index2word[token] for token in x], sequences))
    refined_sentences = [" ".join(seq) for seq in refined_tokens]

    tfidf = TfidfVectorizer(max_features=1000)
    tfidf = tfidf.fit_transform(refined_sentences)
    X = tfidf.toarray()
    print("Vectorizer fitted...")
    
    config = Config('../config')
    model_params = config.load('baseline.json')
    params_dict = model_params["params_dict"]
    print("Params loaded...: ", params_dict)

    model = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    cv = GridSearchCV(model, params_dict, cv=10)
    cv.fit(X, y)

    best_model = cv.best_estimator_

    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    print("Train acc: ",accuracy_score(y_train, y_pred_train))
    print("Test acc: ",accuracy_score(y_test, y_pred_test))