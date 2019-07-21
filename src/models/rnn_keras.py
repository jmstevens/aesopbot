import tensorflow as tf

import numpy as np
import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import codecs
import collections
import math
import re
import tensorflow as tf
from tensorflow.keras.utils import get_file
import io
import string
import csv
from functools import reduce
import random
from tqdm import tqdm
import itertools
from random import shuffle
import re, string, unicodedata
import contractions
import inflect
from bs4 import BeautifulSoup
import pickle
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
import gensim
from copy import copy
from tensorflow.keras.callbacks import EarlyStopping

data_dir = 'data/processed/verses.txt'
with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)
with open(data_dir, "rb") as fp:   # Unpickling
            lyrics = pickle.load(fp)

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

def remove_punctuation( words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

def replace_numbers(words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

def remove_non_ascii( words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

# lyrics = [remove_non_ascii(i) for i in lyrics]
# lyrics = [replace_numbers(i) for i in lyrics]
# lyrics = [remove_punctuation(i) for i in lyrics]

lyrics = [''.join(i) for i in lyrics]
text = ' \n '.join(lyrics)
vocab = sorted(set(text))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100

examples_per_epoch = len(text)//seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
            batch_input_shape=[batch_size, None]),
                rnn(rnn_units,
                    return_sequences=True,
                    recurrent_initializer='glorot_uniform',
                    stateful=True),
        tf.keras.layers.Dense(vocab_size)
        ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model = build_model(
            vocab_size = len(vocab),
            embedding_dim=embedding_dim,
            rnn_units=1024,
            batch_size=BATCH_SIZE
            )

model.compile(
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
    loss = loss)
early_stopping_monitor = EarlyStopping(patience=2, monitor='loss')
history = model.fit(dataset.repeat(), epochs=25, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback, early_stopping_monitor])
