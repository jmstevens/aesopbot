from __future__ import print_function
#import tensorflow.keras library
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import gensim
from gensim.models.doc2vec import LabeledSentence

#import spacy, and spacy french model
# spacy is used to work on text
import spacy
nlp = spacy.load('en')

#import other libraries
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle

sequence_length = 30
sequences_step = 1
sequence_length = 30

rnn_size = 256
learning_rate = 0.001

batch_size = 32
num_epochs = 50
save_dir = 'src/data/tensorboard'

def create_wordlist(doc):
    wl = []
    for word in doc:
        if word.text not in ("\n","\n\n", '\u2009', '\xa0'):
            wl.append(word.text.lower())
    return wl


wordlist = []

with codecs.open('data/processed/lyrics.txt', "r") as file:
    data = tf.compat.as_str(file.read())

doc = nlp(data)
wordlist = create_wordlist(doc)
## Create Dictionary
word_counts = collections.Counter(wordlist)

# Mapping from index to word: that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = {x[0] for x in word_counts.most_common()}

# Get Vocab Size
vocab_size = len(words)
print("vocab size: {}".format(vocab_size))

with open(os.path.join('src','data','vocabulary.pkl'), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

## Need to make two lists
# Sequences: list contains the sequences of words, used to train the model
# next_words: list contains the next words for each sequences of the sequences list
# Create sequences
sequences = []
next_words = []
for i in range(0, len(wordlist) - sequence_length, sequences_step):
    sequences.append(wordlist[i: i + sequence_length])
    next_words.append(wordlist[i + sequence_length])

print('nb sequences:{}'.format(len(sequences)))


# X : the matrix of the following dimensions:
# number of sequences,
# number of words in sequences,
# number of words in the vocabulary.

# y : the matrix of the following dimensions:
# number of sequences,
# number of words in the vocabulary.

X = np.zeros((len(sequences), sequence_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1

def bidirectional_lstm_model(sequence_length, vocab_size, rnn_size, learning_rate):
    print('Building LSTM model')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation='relu'), input_shape=(sequence_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    callbacks = [EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("Model Built!!!")
    return model

def train(model, batch_size, save_dir):
    md = bidirectional_lstm_model(sequence_length, vocab_size, rnn_size, learning_rate)
    md.summary()

    callbacks = [EarlyStopping(patience=4, monitor='val_loss'),
                 ModelCheckpoint(filepath=save_dir + "/" + "my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor='val_loss', verbose=0, mode='auto', period=2)]
    # Fit the model
    history = md.fit(X, y,
                     batch_size=batch_size,
                     shuffle=False,
                     epochs=num_epochs,
                     callbacks=callbacks,
                     shuffle=True,
                     validation_split=0.1)
    md.save(save_dir + "/" + "my_model_gen_sentences.h5")


# train(batch_size, num_epochs)

def generate(save_dir):
    print("Loading vocabulary.........")
    vocab_file = os.path.join('src','data','vocabulary.pkl')

    with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

    vocab_size = len(words)
    print("loading model........")
    model = load_model(os.path.join(save_dir, "my_model_gen_sentences.h5")


# Sentence function
def create_sentences(doc):
    punctuation = [".","?","!",":","...","\n"]
    sentences = []
    sent = []

    for word in doc:
        if word.text not in punctuation:
            if word.text not in ('\u2009','\xa0'):
                sent.append(word.text.lower())
        else:
            sent.append(word.text.lower())
            if len(sent) > 1:
                sentences.append(sent)
            sent=[]
    return sentences

# Create sentences
sentences = create_sentences(doc)
sentences_label = ["ID" + str(i) for i in range(np.array(sentences).shape[0])]


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])




def train_doc2vec_model(data, docLabels, size=300, sample=0.00001, dm=0, hs=1, window=10, min_count=0, workers=8, min_alpha=0.024, epoch=15, save_file='./data/doc2vec.w2v'):
    startime = time.time()

    print("{0} articles loaded for model".format(len(data)))

    it = LabeledLineSentence(data, docLabels)
    model = gensim.models.Doc2Vec(size=size, sample=sample, dm=dm, window=window, min_count=min_count, workers=workers,alpha=alpha, min_alpha=min_alpha, hs=hs) # use fixed learning rate

    model.build_vocab(it)

    for epoch in range(epoch):
        print("Training epoch {}".format(epoch + 1))
        model.train(it,total_examples=model.corpus_count,epochs=model.iter)

    model.save(os.path.join(save_file))
    print('model saved')

train_doc2vec_model(sentences, sentences_label, size=500,sample=0.0,alpha=0.025, min_alpha=0.001, min_count=0, window=10, epoch=20, dm=0, hs=1, save_file='./data/doc2vec.w2v')
