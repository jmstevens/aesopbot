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

# word_vectors = api.load("glove-wiki-gigaword-100")
# class Provider():
#     embedding_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
#     data_dir = 'data/processed/verses.txt'
#     with open('configs/config.json','r') as cfgFile:
#         cfg = json.load(cfgFile)
#
#     def __init__(self, batch_size, sequence_length):
#         with open(self.data_dir, "rb") as fp:   # Unpickling
#             lyrics = pickle.load(fp)
#         lyrics = [self.remove_non_ascii(i) for i in lyrics]
#         lyrics = [self.replace_numbers(i) for i in lyrics]
#         lyrics = [self.remove_punctuation(i) for i in lyrics]
#         lyrics = [''.join(i) for i in lyrics]
#
#         tokenizer = Tokenizer(num_words = 20000)
#         tokenizer.fit_on_texts(lyrics)
#         word_index = tokenizer.word_index
#
#         sequences = tokenizer.texts_to_sequences(lyrics)
#         total_words = len(tokenizer.word_index) + 1
#
#         nb_words = min(20000, len(word_index))+1
#         embedding_matrix = np.zeros((nb_words, 300))
#         for word, i in word_index.items():
#             if word in self.embedding_model.vocab:
#                 embedding_matrix[i] = self.embedding_model.word_vec(word)
#         self.embedding_matrix = embedding_matrix
#
#
#         x = np.array(lyrics)
#         # tokenizer = Tokenizer(num_words = 20000)
#         np.random.shuffle(x)
#         split_size = round(x.shape[0] * float(0.1))
#         train = x[split_size:]
#         test = x[:split_size]
#         #
#         # tokenizer.fit_on_texts(X + Y)
#         #
#         # train_sequences = tokenizer.texts_to_sequences(X)
#         # test_sequences = tokenizer.texts_to_sequences(Y)
#         # word_index = tokenizer.word_index
#
#         def prep(lyrics):
#             input_sequences = []
#             for line in lyrics:
#                 token_list = tokenizer.texts_to_sequences([line])[0]
#                 for i in range(1, len(token_list)):
#                 	n_gram_sequence = token_list[:i+1]
#                 	input_sequences.append(n_gram_sequence)
#             # pad sequences
#             max_sequence_len = max([len(x) for x in input_sequences])
#             input_sequences = np.array(pad_sequences(input_sequences, maxlen=301, padding='pre'))
#
#             # create predictors and label
#             inputs, targets = input_sequences[:,:-1],input_sequences[:,-1]
#             return inputs, targets
#             # labels = ku.to_categorical(self.targets, num_classes=total_words)
#
#         self.train_x, self.train_y = prep(train)
#         self.test_x, self.test_y = prep(test)
#
#         tmp = list(self.factors(self.train_x.shape[0]))
#         tmp.sort()
#         split_num = self.findMiddle(tmp)
#
#         self.input_batches = np.split(self.train_x, split_num)
#         self.target_batches = np.split(self.train_y, split_num)
#         # tmp = list(self.factors(self.test_x.shape[0]))
#         # tmp.sort()
#         # split_num_test = self.findMiddle(tmp.sort())
#         # self.input_batches_test = np.split(self.test_x, split_num_test)
#         # self.target_batches_test = np.split(self.test_y, split_num_test)
#
#     def next_batch(self):
#         inputs = self.input_batches[self.pointer]
#         targets = self.target_batches[self.pointer]
#         self.pointer += 1
#         return inputs, targets
#
#     def next_batch_test(self):
#         inputs = self.input_batches_test[self.pointer_test]
#         targets = self.target_batches_test[self.pointer_test]
#         self.pointer_test += 1
#         return inputs, targets
#
#     def reset_batch_pointer(self):
#             self.pointer = 0
#
#     def reset_batch_pointer_test(self):
#             self.pointer_test = 0
class Provider():
    data_dir = 'data/processed/verses.txt'
    with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)
    def __init__(self, batch_size, sequence_length):
        with open(self.data_dir, "rb") as fp:   # Unpickling
            lyrics = pickle.load(fp)
        # lyrics = [''.join(i).replace("<eol>","\").replace("<eov"," ") for i in lyrics]
        lyrics = [self.remove_non_ascii(i) for i in lyrics]
        lyrics = [self.replace_numbers(i) for i in lyrics]
        lyrics = [self.remove_punctuation(i) for i in lyrics]
        lyrics = [''.join(i) for i in lyrics]
        lyrics = [i.replace("eol","\n").replace("eov","\n\n") for i in lyrics]
        shuffle(lyrics)
        self.lyrics = lyrics
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.pointer = 0
        # self.new_sequence()
        # count_pairs = sorted(collections.Counter(' '.join(self.lyrics).split()).items(), key=lambda x: -x[1])
        count_pairs = sorted(collections.Counter(list(' '.join([' '.join(i) for i in self.lyrics]))).items(), key=lambda x: -x[1])
        self.count_pairs = count_pairs
        self.pointer = 0
        # data = ' '.join([' '.join(i) for i in self.lyrics])
        data = ' '.join(self.lyrics)
        # self.data = data
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

    @staticmethod
    def findMiddle(input_list):
        middle = float(len(input_list))/2
        if middle % 2 != 0:
            return input_list[int(middle - .5)]
        else:
            return input_list[int(middle-1)]

    @staticmethod
    def factors(n):
        return set(reduce(list.__add__,
                    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


class Data(object):
    data_dir = 'data/processed/verses.txt'
    with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)

    def __init__(self):
        with open(self.data_dir, "rb") as fp:   # Unpickling
            lyrics = pickle.load(fp)

        lyrics = [self.remove_non_ascii(i) for i in lyrics]
        lyrics = [self.replace_numbers(i) for i in lyrics]
        lyrics = [self.remove_punctuation(i) for i in lyrics]
        lyrics = [''.join(i).replace("eol","<eol>").replace("eov","<eov") for i in lyrics]
        lyrics = [i for i in lyrics if len(i.split()) <= 300]
        self.lyrics = lyrics

        count_pairs = sorted(collections.Counter(' '.join(self.lyrics).split()).items(), key=lambda x: -x[1])
        data = self.lyrics
        self.chars, _ = zip(*count_pairs)
        self.vocabulary_size = len(self.chars)
        self.vocabulary = dict(zip(self.chars, range(len(self.chars))))

        tokenizer = Tokenizer()

        # basic cleanup
        corpus = data

        # tokenization
        tokenizer.fit_on_texts(corpus)
        total_words = len(tokenizer.word_index) + 1

        # create input sequences using list of tokens
        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
            	n_gram_sequence = token_list[:i+1]
            	input_sequences.append(n_gram_sequence)

        # pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=301, padding='pre'))

        # create predictors and label
        self.inputs, self.targets = input_sequences[:,:-1],input_sequences[:,-1]
        self.labels = ku.to_categorical(self.targets, num_classes=total_words)

        self.input_batches = np.split(self.inputs, 36)
        self.target_batches = np.split(self.targets, 36)

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
