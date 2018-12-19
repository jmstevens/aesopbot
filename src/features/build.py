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

# word_vectors = api.load("glove-wiki-gigaword-100")

class Dataset:
    data_dir = 'data/raw'
    def __init__(self):
        self.download()
        self.transform()
        self.vocab_df()
        self.filtered_artists()

    def download(self):
        print("Downloading file from s3")
        path = get_file(
                'lyrics.csv',
                origin='https://s3.amazonaws.com/hiphopbot/data/lyrics.csv'
        )
        print("Reading downloaded file")
        df = pd.read_csv(path)
        df = df[df['genre'] == 'Hip-Hop']
        print("Transforming dataset")
        value = df['lyrics'].apply(lambda x: self.transform_corpus(x))
        df.loc[:,'lyrics_transform'] = value
        df = df[df['lyrics_transform'] != 'nan']
        self.df = df


    def transform(self):
        print("Transforming file......")
        df = self.df
        print('Editing text.......')
        df = df[df['lyrics_transform'].str.len() > 0]
        df.loc[:, 'song_length'] = df['lyrics_transform'].apply(lambda x: self.song_length(x))
        df.loc[:, 'unique_words'] = df['lyrics_transform'].apply(lambda x: self.count_vocabulary_per_song(x))
        self.df = df
        # return df

    @staticmethod
    def transform_corpus(texts, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.
        # Convert words to lower case and split them
        if type(texts) is not 'str':
            texts = str(texts)
        text_l = []
        for text in texts.lower().split('\n'):
            if 'verse' not in text or len(text.split()) > 3:
                text = re.sub('(ass)+(\w+)--(\w)', 'asshole', text)
                text = re.sub('(f)--(k)', 'fuck', text)
                text = re.sub('(s)--(t)', 'shit', text)
                text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
                text = text.lstrip().rstrip()
                text = re.sub(r":", "", text)
                text = re.sub(r"\s{2,}", " ", text)
                text = re.sub("chorus", " ", text)
                text = re.sub("aesop rock"," ", text)
                text = re.sub(r'^.*: ', '', text)
                text = re.sub(r'[0-9]x','', text)
                text = text.lstrip().rstrip()
                text_l.append(text)

        text = "\n".join(text_l)
        # text.replace("\n", " <nl> ")
        # Return a list of words
        return text.lstrip().rstrip()

    @staticmethod
    def song_length(w):
        return len(set([i for i in w]))

    @staticmethod
    def count_vocabulary_per_song(w):
        return len(re.sub('[,\.!?]','', w).split(' '))

    @staticmethod
    def count_total_vocab_for_artist(df):
        _dict = {}
        for group, value in df.groupby(['artist']):
            foo = re.sub('[\']','',re.sub('[,\.!?"]','',' '.join([i for i in value['lyrics_transform'].tolist()]))).split(' ')
            _dict[group] = len(sorted(list(set(foo))))
        return _dict

    def vocab_df(self):
        df = self.df
        vocab = self.count_total_vocab_for_artist(df)
        high_vocab_keys = []
        high_vocab_items = []
        average_words_per_song = df.groupby('artist')['unique_words'].mean().reset_index()
        for key, value in vocab.items():
            high_vocab_keys.append(key)
            high_vocab_items.append(value)
        df.groupby('artist')['unique_words'].mean().reset_index()['unique_words']
        df.groupby('artist').count()
        vocab_df = pd.DataFrame.from_dict({'artist': high_vocab_keys,
                                           'number_of_words': high_vocab_items,
                                           'average_words_per_song': df.groupby('artist')['unique_words'].mean().reset_index()['unique_words'],
                                           'total_songs': df.groupby('artist')['unique_words'].count().reset_index()['unique_words'],
                                           })
        vocab_df['words_by_song'] = vocab_df['number_of_words']/vocab_df['total_songs']
        vocab_df['vocab_metric'] = (vocab_df['number_of_words'] * vocab_df['average_words_per_song']) / vocab_df['total_songs']
        self.vocab_df = vocab_df
        return vocab_df

    def filtered_artists(self):
        vocab_df = self.vocab_df
        filtered_artists = vocab_df[vocab_df['total_songs'] > 25].sort_values(['vocab_metric']).nlargest(25,'vocab_metric')['artist'].tolist()
        print(filtered_artists)
        filtered_artists_df = self.df[self.df['artist'].isin(filtered_artists)]
        self.df_filtered = filtered_artists_df
        return filtered_artists_df

    def save_artist(self, artist=None):
        if artist:
            _df = self.df_filtered
            df = _df[_df['artist'] == artist]
            df.to_csv('data/processed/songs/lyrics.csv')
            df = df.groupby('song')
        else:
            self.df_filtered.to_csv('data/processed/songs/lyrics.csv')
            df = self.df_filtered.groupby('song')
        for group, value in df:
            song_text = ' '.join(value['lyrics_transform'])
            if len(song_text) > 3:
                print('Saving lyrics for song: {}'.format(group))
                with open("data/processed/songs/{}_lyrics.txt".format(group), "w", encoding="utf-8") as text_file:
                    text_file.write(song_text)
                    text_file.close()

class Provider(Dataset):
    data_dir = 'src/data/'
    def __init__(self, batch_size, sequence_length):
        # Processing().save(t)
        self.dataset = pd.read_csv('data/processed/songs/lyrics.csv')
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.lyrics = []
        self.load()
        # self.tokens = self.get_vocabulary()
        self.num_songs = len(self.dataset['song'].tolist())
        self.pointer = 0
        # self.new_sequence()

        # count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
        count_pairs = sorted(collections.Counter(' '.join([' '.join(i) for i in self.lyrics]).split(' ')).items(), key=lambda x: -x[1])
        self.pointer = 0
        data = ' '.join([' '.join(i) for i in self.lyrics]).split(' ')
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

    def load(self):
        files = os.listdir('data/processed/songs/')
        files = [i for i in files if '.txt' in i]
        self.songs_index = {v: k for v, k in enumerate(files)}
        songs_copy = list(self.songs_index.keys())
        # shuffle(songs_copy)
        for file in songs_copy:
            with codecs.open('data/processed/songs/{}'.format(self.songs_index[file]), "r", encoding="utf-8") as song:
                song_lyrics = song.read().split(" ") #.replace(" \n "," <nl> ")
                self.lyrics.append(song_lyrics)
        # self.lyrics = ' '.join([' '.join(i) for i in self.lyrics])
    # def clean(self, string):
    #     string = string.lower()  # lowercase
    #
    #     clean_words = []
    #     for word in string.split():
    #         # clean words with quotation marks on only one side
    #         if word[0] == '"' and word[-1] != '"':
    #             word = word[1:]
    #         elif word[-1] == '"' and word[0] != '"':
    #             word = word[-1]
    #
    #         # clean words with parenthases on only one side
    #         if word[0] == '(' and word[-1] != ')':
    #             word = word[1:]
    #         elif word[-1] == ')' and word[0] != '(':
    #             word = word[:-1]
    #
    #         clean_words.append(word)
    #     clean_words = re.sub('\n','999newline999',' '.join(clean_words))
    #     clean_words = re.sub('[\W_]',' ',clean_words)
    #     return re.sub('newline','<newline>',clean_words)

    # def get_vocabulary(self):
    #     all_words = reduce(lambda a,b: a + b, self.lyrics)
    #     tokens = sorted(list(set(all_words)))
    #
    #     # Create a map from word to index
    #     self.vocab_lookup = dict((word, i) for i, word in enumerate(tokens))
    #     self.lyric_indices = [map(lambda word: self.vocab_lookup[word], song) for song in self.lyrics]
    #
    #     print(len(tokens))
    #
    #     return tokens

    # def new_sequence(self):
    #     tokenizer = Tokenizer(num_words=200000)
    #
    #     tokenizer.fit_on_texts(self.lyrics)
    #
    #     sequences = tokenizer.texts_to_sequences(self.lyrics)
    #     word_index = tokenizer.word_index
    #     print('Found %s unique tokens' % len(word_index))
    #
    #     inputs = pad_sequences(sequences, maxlen=self.sequence_length)
    #     targets = np.copy(inputs)
    #     targets[:-1] = inputs[1:]
    #     targets[-1] = inputs[0]
    #     print('Shape of input data tensor:', inputs.shape)
    #     print('Shape of output data tensor:', targets.shape)

        # self.inputs = inputs
        # self.outputs = targets

    def next_batch(self):
        inputs = self.input_batches[self.pointer]
        targets = self.target_batches[self.pointer]
        self.pointer += 1
        return inputs, targets

    # def get_sequence(self, sequence_length=None):
    #     if not sequence_length:
    #         sequence_length = self.sequence_length
    #     while True:
    #         df = self.dataset.sample(1)
    #         with codecs.open('data/processed/songs/{}_lyrics.txt'.format(df['song'].tolist()[0]), "r", encoding="utf-8") as song:
    #             song_lyrics = song.read().split('\n')
    #             print(song_lyrics)
    #         if (len(song_lyrics) - (sequence_length + 1)) < 0:
    #             pass
    #         else:
    #             i = random.randint(0, len(song_lyrics) - (sequence_length + 1))
    #             input = np.array([self.vocab_lookup[i] for i in song_lyrics[i:i + sequence_length]], dtype=int)
    #             target = np.array([self.vocab_lookup[i] for i in song_lyrics[i + 1:i + sequence_length + 1]], dtype=int)
    #             break
    #     return input, target

    def reset_batch_pointer(self):
            self.pointer = 0

# _b = Provider(50, 32)
# print(_b.get_train_batch()[0].shape)

    #
    #     self.data = data
    #     count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
    #     self.pointer = 0
    #     self.chars, _ = zip(*count_pairs)
    #     self.vocabulary_size = len(self.chars)
    #     self.vocabulary = dict(zip(self.chars, range(len(self.chars))))
    #     self.tensor = np.array(list(map(self.vocabulary.get, data)))
    #     self.batches_size = int(self.tensor.size / (self.batch_size * self.sequence_length))
    #
    #     if self.batches_size == 0:
    #         assert False, "Unable to generate batches. Reduce size or sequence length"
    #
    #     self.tensor = self.tensor[:self.batches_size * self.batch_size * self.sequence_length]
    #     inputs = self.tensor
    #     targets = np.copy(self.tensor)
    #
    #     targets[:-1] = inputs[1:]
    #     targets[-1] = inputs[0]
    #
    #     self.input_batches = np.split(inputs.reshape(self.batch_size, -1), self.batches_size, 1)
    #     self.target_batches = np.split(targets.reshape(self.batch_size, -1), self.batches_size, 1)
    #
    #     print("Tensor size: " + str(self.tensor.size))
    #     print("Batch size: " + str(self.batch_size))
    #     print("Sequence length: " + str(self.sequence_length))
    #     print("Batches size: " + str(self.batches_size))
    #     print("")
    #
    # def next_batch(self):
    #     inputs = self.input_batches[self.pointer]
    #     targets = self.target_batches[self.pointer]
    #     self.pointer += 1
    #     return inputs, targets
    #
    # def reset_batch_pointer(self):
    #     self.pointer = 0
    #
    # # @staticmethod
    # # def build_dataset(words, n_words):
    # #     """Process raw inputs into a dataset."""
    #     count = [['UNK', -1]]
    #     count.extend(collections.Counter(words).most_common(n_words - 1))
    #     dictionary = dict()
    #     for word, _ in count:
    #         dictionary[word] = len(dictionary)
    #     data = list()
    #     unk_count = 0
    #     for word in words:
    #         if word in dictionary:
    #             index = dictionary[word]
    #         else:
    #             index = 0  # dictionary['UNK']
    #             unk_count += 1
    #         data.append(index)
    #     count[0][1] = unk_count
    #     reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #     return data, count, dictionary, reversed_dictionary
    #
    # # @staticmethod
    # def generate_batch(self, batch_size, num_skips, skip_window):
    #     # global data_index
    #     assert batch_size % num_skips == 0
    #     assert num_skips <= 2 * skip_window
    #     data_index = self.data_index
    #     batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    #     labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    #     span = 2 * skip_window + 1
    #     buffer = collections.deque(maxlen=span)
    #
    #     for _ in range(span):
    #         buffer.append(self.data[data_index])
    #         data_index = (data_index + 1) % len(self.data)
    #
    #     for i in range(batch / num_skips):
    #         target = skip_window
    #         targets_to_avoid = [skip_window]
    #         for j in range(num_skips):
    #             while target in targets_to_avoid:
    #                 target = random.randint(0, span - 1)
    #             target_to_avoid.append(target)
    #             batch[i * num_skips + j] = buffer[skip_window]
    #             labels[i * num_skips + j, 0] = buffer[target]
    #         buffer.append(data[data_index])
    #         data_index = (data_index + 1) % len(data)
    #     data_index = (data_index + len(data) - span) % len(data)
    #     return batch, label
