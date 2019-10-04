import tensorflow as tf
import pandas as pd
import numpy as np
import os
import codecs
import collections
import math
import re
from tensorflow.keras.utils import get_file
import io
import string
import csv
from functools import reduce
import random
import itertools
from random import shuffle
import re, string, unicodedata
import inflect
import pickle
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from copy import copy
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

class Lyrics():
    BUFFER_SIZE = 1024

    with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)
    data_dir = 'data/processed/verses.txt'
    with open(data_dir, "rb") as fp:   # Unpickling
        lyrics = pickle.load(fp)

    def __init__(self, BATCH_SIZE, word_based=True):
        self.BATCH_SIZE = BATCH_SIZE
        self.word_based = word_based

    def build_words(self, train_split=0.2):
        print("Removing non-ascii.....")
        lyrics = [self.remove_non_ascii(i) for i in self.lyrics]
        print("Replacing numbers.....")
        lyrics = [self.replace_numbers(i) for i in lyrics]
        print("Removing punctuation.....")
        lyrics = [self.remove_punctuation(i) for i in lyrics]
        print("Joining back lyrics.....")
        lyrics = ' '.join([''.join(i) for i in lyrics])
        tokens = nltk.WhitespaceTokenizer().tokenize(lyrics)
        # stop = set(stopwords.words('english'))
        # clean_tokens = [tok for tok in tokens if tok not in stop]
        freq_dist = nltk.FreqDist(tokens)
        rarewords = list(freq_dist.keys())[-10000:]
        after_rare_words = [word for word in tokens if word not in rarewords]
        words = []
        for word in after_rare_words:
            if word == 'eol':
                word = '\n'
            words.append(word)
        # verses = [nltk.WhitespaceTokenizer().tokenize(verse) for verse in lyrics]
        # lemmatizer = nltk.WordNetLemmatizer()
        # lemma = [lemmatizer.lemma tize(word, pos='v') for word in words]
        # texts = list(itertools.chain.from_iterable(lemma))
        # freq_dist = nltk.FreqDist(texts)    rarewords = freq_dist.keys()[-5:]
        # words = lemma
        vocab = sorted(set(words))
        vocab_size = len(vocab)
        word2idx = {u:i for i, u in enumerate(vocab)}
        idx2word = np.array(vocab)

        verse_data = [word2idx[c] for c in words]
        # text_in_words = [w for w in ' '.join(lyrics).split(' ') if w.strip()]
        # word_freq = {}
        # for word in text_in_words:
        #     word_freq[word] = word_freq.get(word, 0) + 1
        # ignored_words = set()
        # for k, v in word_freq.items():
        #     if word_freq[k] >= 10:
        #         ignored_words.add(k)
        #
        # ignored_words = list(set(ignored_words))
        # print(len(ignored_words))
        # word_index = {k:(v+1) for k,v in word2idx.items()}
        # vocab_size = len(word_index)
        # word_index['<PAD>'] = 0
        # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        # max_len = int(np.asarray([len(verse.split(' ')) for verse in data_full_words.split('<EOV>')]).mean()) + int(np.ceil(np.asarray([len(verse.split(' ')) for verse in data_full_words.split('<EOV>')]).std()))
        # data_full_words = ' '.join(lyrics)
        # data_verses = []
        # data_verses_idx = []
        # for verse in data_full_words.split('<EOV>'):
        #     verse_tmp = []
        #     verses_idx = []
        #     verse = verse + ' <EOV>'
        #     if len(verse.split(' ')) <= max_len:
        #         for word in verse.split(' '):
        #             if word != '' and word not in set(ignored_words):
        #                 verses_idx.append(word_index[word])
        #                 verse_tmp.append(word)
        #     if verse_tmp:
        #         data_verses.append(verse_tmp)
        #     if verses_idx:
        #         data_verses_idx.append(verses_idx)
        #
        # split_size = int(np.ceil(len(data_verses_idx) * train_split))
        # #max_len = 12
        # verse_data_padded = tf.keras.preprocessing.sequence.pad_sequences(data_verses_idx,
        #                                                                   value=word_index['<PAD>'],
        #                                                                   padding='pre',
        #                                                                   maxlen=max_len+1)
        # char_dataset = tf.data.Dataset.from_tensor_slices(verse_data)
        # dataset = char_dataset.map(self.split_input_target)

        # train_data = dataset.skip(split_size).shuffle(self.BUFFER_SIZE)
        # dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)

        char_dataset = tf.data.Dataset.from_tensor_slices(verse_data)
        sequences = char_dataset.batch(256, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)
        dataset = dataset.shuffle(10000).batch(32, drop_remainder=True)
        # test_data = dataset.take(split_size)
        # test_data = test_data.batch(self.BATCH_SIZE)



        return dataset, word2idx, idx2word, vocab_size

    def build_char(self, max_len=100):
        print("Removing non-ascii.....")
        lyrics = [self.remove_non_ascii(i) for i in self.lyrics]
        print("Replacing numbers.....")
        # lyrics = [self.replace_numbers(i) for i in lyrics]
        print("Removing punctuation.....")
        lyrics = [self.remove_punctuation(i) for i in lyrics]
        # print("Joining back lyrics.....")
        lyrics = [''.join(i) for i in lyrics]
        lyrics = [i.replace("eol","<eol>").replace("eov","<eov>") for i in lyrics]
        # lyrics = self.lyrics
        # lyrics = [i.replace("eol","<eol>").replace("eov","<eov>") for i in lyrics]
        data_full_words = ''.join(lyrics)
        data_chars = [i for i in data_full_words]

        self.vocab = sorted(set(' '.join(lyrics)))
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        data_verses_idx = [self.char2idx[i] for i in data_chars]

        char_dataset = tf.data.Dataset.from_tensor_slices(data_verses_idx)
        sequences = char_dataset.batch(max_len, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)

        return dataset

    @staticmethod
    def remove_non_ascii(words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    @staticmethod
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

    @staticmethod
    def remove_punctuation(words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    @staticmethod
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text


#
#
#
# class Dataset():
#     BATCH_SIZE = 64
#     BUFFER_SIZE = 50000
#     data_dir = 'data/processed/verses.txt'
#     with open('configs/config.json','r') as cfgFile:
#         cfg = json.load(cfgFile)
#
#     def __init__(self, batch_size):
#         with open(self.data_dir, "rb") as fp:   # Unpickling
#             lyrics = pickle.load(fp)
#
#         lyrics = [self.remove_non_ascii(i) for i in lyrics]
#         # lyrics = [self.replace_numbers(i) for i in lyrics]
#         # lyrics = [self.remove_punctuation(i) for i in lyrics]
#         lyrics = [''.join(i) for i in lyrics]
#         lyrics = [i.replace("eol"," \n ").replace("eov"," \n\n ") for i in lyrics]
#         lyrics = [i.replace(' < \n >', ' <eol>').replace(' < \n\n >',' <eov>') for i in self.lyrics]
#         text_in_words = [w for w in ''.join(lyrics).split(' ') if w.strip() != '' or w == '\n']
#         # word_freq = {}
#         # for word in text_in_words:
#         #     word_freq[word] = word_freq.get(word, 0) + 1
#         # ignored_words = set()
#         # for k, v in word_freq.items():
#         #     if word_freq[k] <= 3:
#         #         ignored_words.add(k)
#         # ignored_words = list(ignored_words)
#         vocab = sorted(set('  '.join(lyrics).split(' ')))
#         word2idx = {u:i for i, u in enumerate(vocab)}
#         idx2word = np.array(vocab)
#
#         # Shuffle lyrics
#         shuffle(lyrics)
#
#         data_full_words = ' '.join(lyrics)
#         data_verses = [i + ' <eol> ' for i in data_full_words.split(' <eol> ')]
#         word_index = {k:(v+1) for k,v in word2idx.items()}
#         word_index['<PAD>'] = 0
#         reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#         max_len = np.asarray([len(i) for i in data_verses]).max()
#
#         data_full_idx = [[word_index[k] for k in i.split(' ')] for i in data_full_words]
#         data_verses_idx = [[word_index[k] for k in i.split(' ')] for i in data_verses]
#         verse_data_padded = tf.keras.preprocessing.sequence.pad_sequences(data_verses_idx, value=word_index['<PAD>'], padding='post', maxlen=max_len)
#
#         char_dataset = tf.data.Dataset.from_tensor_slices(verse_data_padded)
#         sequences = char_dataset.batch(max_len+1, drop_remainder=True)
#         dataset = sequences.map(self.split_input_target)
#         ratio_train = np.ceil(len(verses) * .8)
#         ratio_test = np.ceil(len(verses) - ratio_train) - 1
#
#         # all_labeled_data = dataset.shuffle(self.BUFFER_SIZE, reshuffle_each_iteration=False)
#
#         train_data = all_labeled_data.skip(ratio_train).shuffle(self.BUFFER_SIZE)
#         self.train_data = train_data.batch(self.BATCH_SIZE)
#
#         test_data = all_labeled_data.take(ratio_test)
#         self.test_data = test_data.batch(self.BATCH_SIZE)
#
#     @staticmethod
#     def split_input_target(chunk):
#         input_text = chunk[:-1]
#         target_text = chunk[1:]
#         return input_text, target_text
#
#     @staticmethod
#     def replace_numbers(words):
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
#     @staticmethod
#     def remove_punctuation(words):
#         """Remove punctuation from list of tokenized words"""
#         new_words = []
#         for word in words:
#             new_word = re.sub(r'[^\w\s]', '', word)
#             if new_word != '':
#                 new_words.append(new_word)
#         return new_words
#
#
#
# class Data(object):
#     data_dir = 'data/processed/verses.txt'
#     with open('configs/config.json','r') as cfgFile:
#         cfg = json.load(cfgFile)
#
#     def __init__(self):
#         with open(self.data_dir, "rb") as fp:   # Unpickling
#             lyrics = pickle.load(fp)
#
#         lyrics = [self.remove_non_ascii(i) for i in lyrics]
#         lyrics = [self.replace_numbers(i) for i in lyrics]
#         lyrics = [self.remove_punctuation(i) for i in lyrics]
#         lyrics = [''.join(i).replace("eol","<eol>").replace("eov","<eov>") for i in lyrics]
#         lyrics = [i for i in lyrics if len(i.split()) <= 300]
#         self.lyrics = lyrics
#
#         count_pairs = sorted(collections.Counter(' '.join(self.lyrics).split()).items(), key=lambda x: -x[1])
#         data = self.lyrics
#         self.chars, _ = zip(*count_pairs)
#         self.vocabulary_size = len(self.chars)
#         self.vocabulary = dict(zip(self.chars, range(len(self.chars))))
#
#         tokenizer = Tokenizer()
#
#         # basic cleanup
#         corpus = data
#
#         # tokenization
#         tokenizer.fit_on_texts(corpus)
#         total_words = len(tokenizer.word_index) + 1
#
#         # create input sequences using list of tokens
#         input_sequences = []
#         for line in corpus:
#             token_list = tokenizer.texts_to_sequences([line])[0]
#             for i in range(1, len(token_list)):
#             	n_gram_sequence = token_list[:i+1]
#             	input_sequences.append(n_gram_sequence)
#
#         # pad sequences
#         max_sequence_len = max([len(x) for x in input_sequences])
#         input_sequences = np.array(pad_sequences(input_sequences, maxlen=301, padding='pre'))
#
#         # create predictors and label
#         self.inputs, self.targets = input_sequences[:,:-1],input_sequences[:,-1]
#         self.labels = ku.to_categorical(self.targets, num_classes=total_words)
#
#         self.input_batches = np.split(self.inputs, 36)
#         self.target_batches = np.split(self.targets, 36)
#
#     def next_batch(self):
#         inputs = self.input_batches[self.pointer]
#         targets = self.target_batches[self.pointer]
#         self.pointer += 1
#         return inputs, targets
#
#     def reset_batch_pointer(self):
#             self.pointer = 0
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
#     def remove_punctuation(self, words):
#         """Remove punctuation from list of tokenized words"""
#         new_words = []
#         for word in words:
#             new_word = re.sub(r'[^\w\s]', '', word)
#             if new_word != '':
#                 new_words.append(new_word)
#         return new_words
#
#
#
# #
# #
# # class Dataset:
# #     data_dir = 'data/raw'
# #     def __init__(self):
# #         self.download()
# #         # self.transform()
# #         # self.vocab_df()
# #         # self.filtered_artists()
# #
# #     def download(self):
# #         print("Downloading file from s3")
# #         path = get_file(
# #                 'lyrics.csv',
# #                 origin='https://s3.amazonaws.com/hiphopbot/data/lyrics.csv'
# #         )
# #         print("Reading downloaded file")
# #         df = pd.read_csv(path)
# #         df = df[df['genre'] == 'Hip-Hop']
# #         print("Transforming dataset")
# #         value = df['lyrics'].apply(lambda x: self.normalize(x))
# #         df.loc[:,'lyrics_transform'] = value
# #         # df = df[df['lyrics_transform'] != 'nan']
# #         self.df = df
# #
# #     @staticmethod
# #     def get_verses(song):
# #         verse_lines = list()
# #         if isinstance(song, str):
# #             verse_lines = list()
# #             lines = song.splitlines()
# #             for l in range(len(lines)):
# #                 title = [x.lower() for x in lines[l].replace('[', '').replace(']', '').split()]
# #                 if '[' in lines[l] and 'verse' in title:
# #                     section_lines = []
# #                     count = l + 1
# #                     done = False
# #                     while count < len(lines) and not done:
# #                         if '[' not in lines[count]:
# #                             if lines[count] != '':
# #                                 section_lines.append(lines[count])
# #                             count += 1
# #                         else:
# #                             done = True
# #                     self.normalize(section_lines)
# #                     verse_lines.append(section_lines)
# #         return verse_lines
# #
# #
# #     def transform(self):
# #         print("Transforming file......")
# #         df = self.df
# #         print('Editing text.......')
# #         df = df[df['lyrics_transform'].str.len() > 0]
# #         df.loc[:, 'song_length'] = df['lyrics_transform'].apply(lambda x: self.song_length(x))
# #         df.loc[:, 'unique_words'] = df['lyrics_transform'].apply(lambda x: self.count_vocabulary_per_song(x))
# #         self.df = df
# #         # return df
# #
# #     @staticmethod
# #     def transform_corpus(texts, remove_stopwords=False, stem_words=False):
# #         # Clean the text, with the option to remove stopwords and to stem words.
# #         # Convert words to lower case and split them
# #         if type(texts) is not 'str':
# #             texts = str(texts)
# #         text_l = []
# #         for text in texts.lower().split('\n'):
# #             if 'verse' not in text or len(text.split()) > 3:
# #                 text = re.sub('(ass)+(\w+)--(\w)', 'asshole', text)
# #                 text = re.sub('(f)--(k)', 'fuck', text)
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
if __name__ == '__main__':
    _l = Lyrics(64)
    print(_l.build_words(train_split=.1))
