'''

--Preprocess Data and generate features

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

class Config:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
    
    def read_config(self, filename):
        file_path = os.path.join(self.path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            f.close()
        return data

    def save_config(self, data, filename):
        file_path = os.path.join(self.path, filename)
        with open(file_path, 'w') as f:
            json.dump(data, file_path)
            f.close()


def _tokens(sentence):
    return str(sentence).split()


def make_tokens(sentence_list):
    return list(map(_tokens, sentence_list))


def make_vocab(tokens):
    return list(set(chain.from_iterable(tokens)))


def make_word2index(vocab):
    index2word = dict(enumerate(vocab))
    return {value: key for key, value in index2word.items()}


def frequency(tokens):
    return dict(Counter(chain.from_iterable(tokens)))


def _filter_by_count(tokens, vocab):
    return list(filter(lambda x: x in vocab, tokens))


def filter_by_count(tokens, vocab_size):
    counts = frequency(tokens)
    top = sorted(counts.items(), key = lambda x: x[1], reverse = True)[:vocab_size]
    vocab = [word for word, _ in top]
    return list(map(lambda x: _filter_by_count(x, vocab), tokens))


def _sent2index(word2index, tokens):
    return list(map(lambda x: word2index[x], tokens))


def sent2index(word2index, tokens):
    return list(map(lambda x: _sent2index(word2index, x), tokens))



