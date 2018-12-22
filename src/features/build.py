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
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import nltk
from random import shuffle

from gensim.models import KeyedVectors
import gensim.downloader as api
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pickle
import json

# word_vectors = api.load("glove-wiki-gigaword-100")
class Provider():
    data_dir = 'data/processed/verses.txt'
    with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)
    def __init__(self, batch_size, sequence_length):
        with open(self.data_dir, "rb") as fp:   # Unpickling
            lyrics = pickle.load(fp)
        lyrics = [self.remove_non_ascii(i) for i in lyrics]
        lyrics = [self.replace_numbers(i) for i in lyrics]
        lyrics = [self.remove_punctuation(i) for i in lyrics]
        lyrics = [''.join(i).replace("eol","<eol>").replace("eov","<eov") for i in lyrics]
        self.lyrics = lyrics
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.pointer = 0
        # self.new_sequence()
        count_pairs = sorted(collections.Counter(' '.join(self.lyrics).split()).items(), key=lambda x: -x[1])
        # count_pairs = sorted(collections.Counter(' '.join([' '.join(i) for i in self.lyrics]).split(' ')).items(), key=lambda x: -x[1])
        self.pointer = 0
        # data = ' '.join([' '.join(i) for i in self.lyrics]).split(' ')
        data = ' '.join(self.lyrics).split()
        self.chars, _ = zip(*count_pairs)
        self.vocabulary_size = len(self.chars)
        self.vocabulary = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocabulary.get, data)))
        self.batches_size = int(self.tensor.size / (self.batch_size * self.sequence_length))

        if self.batches_size == 0:
            assert False, "Unable to generate batches. Reduce size or sequence length"

        self.tensor = self.tensor[:self.batches_size * self.batch_size * self.sequence_length]
        inputs = self.tensor
        targets = np.copy(self.tensor)

        targets[:-1] = inputs[1:]
        targets[-1] = inputs[0]

        self.input_batches = np.split(inputs.reshape(self.batch_size, -1), self.batches_size, 1)
        self.target_batches = np.split(targets.reshape(self.batch_size, -1), self.batches_size, 1)


    def next_batch(self):
        inputs = self.input_batches[self.pointer]
        targets = self.target_batches[self.pointer]
        self.pointer += 1
        return inputs, targets

    def reset_batch_pointer(self):
            self.pointer = 0

    @classmethod
    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    @classmethod
    def replace_numbers(self, words):
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

    @classmethod
    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

#
#
# class Dataset:
#     data_dir = 'data/raw'
#     def __init__(self):
#         self.download()
#         # self.transform()
#         # self.vocab_df()
#         # self.filtered_artists()
#
#     def download(self):
#         print("Downloading file from s3")
#         path = get_file(
#                 'lyrics.csv',
#                 origin='https://s3.amazonaws.com/hiphopbot/data/lyrics.csv'
#         )
#         print("Reading downloaded file")
#         df = pd.read_csv(path)
#         df = df[df['genre'] == 'Hip-Hop']
#         print("Transforming dataset")
#         value = df['lyrics'].apply(lambda x: self.normalize(x))
#         df.loc[:,'lyrics_transform'] = value
#         # df = df[df['lyrics_transform'] != 'nan']
#         self.df = df
#
#     @staticmethod
#     def get_verses(song):
#         verse_lines = list()
#         if isinstance(song, str):
#             verse_lines = list()
#             lines = song.splitlines()
#             for l in range(len(lines)):
#                 title = [x.lower() for x in lines[l].replace('[', '').replace(']', '').split()]
#                 if '[' in lines[l] and 'verse' in title:
#                     section_lines = []
#                     count = l + 1
#                     done = False
#                     while count < len(lines) and not done:
#                         if '[' not in lines[count]:
#                             if lines[count] != '':
#                                 section_lines.append(lines[count])
#                             count += 1
#                         else:
#                             done = True
#                     self.normalize(section_lines)
#                     verse_lines.append(section_lines)
#         return verse_lines
#
#
#     def transform(self):
#         print("Transforming file......")
#         df = self.df
#         print('Editing text.......')
#         df = df[df['lyrics_transform'].str.len() > 0]
#         df.loc[:, 'song_length'] = df['lyrics_transform'].apply(lambda x: self.song_length(x))
#         df.loc[:, 'unique_words'] = df['lyrics_transform'].apply(lambda x: self.count_vocabulary_per_song(x))
#         self.df = df
#         # return df
#
#     @staticmethod
#     def transform_corpus(texts, remove_stopwords=False, stem_words=False):
#         # Clean the text, with the option to remove stopwords and to stem words.
#         # Convert words to lower case and split them
#         if type(texts) is not 'str':
#             texts = str(texts)
#         text_l = []
#         for text in texts.lower().split('\n'):
#             if 'verse' not in text or len(text.split()) > 3:
#                 text = re.sub('(ass)+(\w+)--(\w)', 'asshole', text)
#                 text = re.sub('(f)--(k)', 'fuck', text)
#                 text = re.sub('(s)--(t)', 'shit', text)
#                 text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
#                 text = text.lstrip().rstrip()
#                 text = re.sub(r":", "", text)
#                 text = re.sub(r"\s{2,}", " ", text)
#                 text = re.sub("chorus", " ", text)
#                 text = re.sub("aesop rock"," ", text)
#                 text = re.sub(r'^.*: ', '', text)
#                 text = re.sub(r'[0-9]x','', text)
#                 text = text.lstrip().rstrip()
#                 text_l.append(text)
#
#         text = "\n".join(text_l)
#         return text.lstrip().rstrip()
#
#     @classmethod
#     def remove_non_ascii(self, words):
#         """Remove non-ASCII characters from list of tokenized words"""
#         new_words = []
#         for word in words:
#             new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#             new_words.append(new_word)
#         return new_words
#
#     @classmethod
#     def to_lowercase(self, words):
#         """Convert all characters to lowercase from list of tokenized words"""
#         new_words = []
#         for word in words:
#             new_word = word.lower()
#             new_words.append(new_word)
#         return new_words
#
#     @classmethod
#     def remove_punctuation(self, words):
#         """Remove punctuation from list of tokenized words"""
#         new_words = []
#         for word in words:
#             new_word = re.sub(r'[^\w\s]', '', word)
#             if new_word != '':
#                 new_words.append(new_word)
#         return new_words
#
#     @classmethod
#     def replace_numbers(self, words):
#         """Replace all interger occurrences in list of tokenized words with textual representation"""
#         p = inflect.engine()
#         new_words = []
#         for word in words:
#             if word.isdigit():
#                 new_word = p.number_to_words(word)
#                 new_words.append(new_word)
#             else:
#                 new_words.append(word)
#         return new_words
#
#     @classmethod
#     def remove_stopwords(self, words):
#         """Remove stop words from list of tokenized words"""
#         new_words = []
#         for word in words:
#             if word not in stopwords.words('english'):
#                 new_words.append(word)
#         return new_words
#
#     @classmethod
#     def stem_words(self, words):
#         """Stem words in list of tokenized words"""
#         stemmer = LancasterStemmer()
#         stems = []
#         for word in words:
#             stem = stemmer.stem(word)
#             stems.append(stem)
#         return stems
#
#     @classmethod
#     def lemmatize_verbs(self, words):
#         """Lemmatize verbs in list of tokenized words"""
#         lemmatizer = WordNetLemmatizer()
#         lemmas = []
#         for word in words:
#             lemma = lemmatizer.lemmatize(word, pos='v')
#             lemmas.append(lemma)
#         return lemmas
#
#     @classmethod
#     def normalize(self, words):
#         if isinstance(words, str):
#             print(words)
#             words = self.remove_non_ascii(words)
#             words = self.to_lowercase(words)
#             words = self.remove_punctuation(words)
#             words = self.replace_numbers(words)
#             words = self.remove_stopwords(words)
#         return words
#
#     @staticmethod
#     def song_length(w):
#         return len(set([i for i in w]))
#
#     @staticmethod
#     def count_vocabulary_per_song(w):
#         return len(re.sub('[,\.!?]','', w).split(' '))
#
#     @staticmethod
#     def count_total_vocab_for_artist(df):
#         _dict = {}
#         for group, value in df.groupby(['artist']):
#             foo = re.sub('[\']','',re.sub('[,\.!?"]','',' '.join([i for i in value['lyrics_transform'].tolist()]))).split(' ')
#             _dict[group] = len(sorted(list(set(foo))))
#         return _dict
#
#     def vocab_df(self):
#         df = self.df
#         vocab = self.count_total_vocab_for_artist(df)
#         high_vocab_keys = []
#         high_vocab_items = []
#         average_words_per_song = df.groupby('artist')['unique_words'].mean().reset_index()
#         for key, value in vocab.items():
#             high_vocab_keys.append(key)
#             high_vocab_items.append(value)
#         df.groupby('artist')['unique_words'].mean().reset_index()['unique_words']
#         df.groupby('artist').count()
#         vocab_df = pd.DataFrame.from_dict({'artist': high_vocab_keys,
#                                            'number_of_words': high_vocab_items,
#                                            'average_words_per_song': df.groupby('artist')['unique_words'].mean().reset_index()['unique_words'],
#                                            'total_songs': df.groupby('artist')['unique_words'].count().reset_index()['unique_words'],
#                                            })
#         vocab_df['words_by_song'] = vocab_df['number_of_words']/vocab_df['total_songs']
#         vocab_df['vocab_metric'] = (vocab_df['number_of_words'] * vocab_df['average_words_per_song']) / vocab_df['total_songs']
#         self.vocab_df = vocab_df
#         return vocab_df
#
#     def filtered_artists(self):
#         vocab_df = self.vocab_df
#         filtered_artists = vocab_df[vocab_df['total_songs'] > 25].sort_values(['vocab_metric']).nlargest(25,'vocab_metric')['artist'].tolist()
#         print(filtered_artists)
#         filtered_artists_df = self.df[self.df['artist'].isin(filtered_artists)]
#         self.df_filtered = filtered_artists_df
#         return filtered_artists_df
#
#     def save_artist(self, artist=None):
#         if artist:
#             _df = self.df_filtered
#             df = _df[_df['artist'] == artist]
#             df.to_csv('data/processed/songs/lyrics.csv')
#             df = df.groupby('song')
#         else:
#             self.df_filtered.to_csv('data/processed/songs/lyrics.csv')
#             df = self.df_filtered.groupby('song')
#         for group, value in df:
#             song_text = ' '.join(value['lyrics_transform'])
#             if len(song_text) > 3:
#                 print('Saving lyrics for song: {}'.format(group))
#                 with open("data/processed/songs/{}_lyrics.txt".format(group), "w", encoding="utf-8") as text_file:
#                     text_file.write(song_text)
#                     text_file.close()
#
#
#
#
