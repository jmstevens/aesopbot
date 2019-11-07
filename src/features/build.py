import tensorflow as tf
# import tensorflow_text as text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import json
import tensorflow_datasets as tfds
import pickle


class Lyrics:
    BUFFER_SIZE = 10000
    with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)

    def __init__(self, BATCH_SIZE, VOCAB_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.VOCAB_SIZE = VOCAB_SIZE
        # self.tokenizer = RegexpTokenizer(r'\w')
        data_dir = 'data/processed/verses.txt'
        with open(data_dir, "rb") as fp:   # Unpickling
            lyrics = pickle.load(fp)
        lyrics = np.array(lyrics)
        arr = [[j for j in i.split(' \n ') if len(j) > 1 and '\n\n' != j] for i in list(np.array(lyrics)) if len(i.split(' \n ')) > 0]
        flattened_list = np.asarray([y for x in arr for y in x])
        print(flattened_list)
        self.target = flattened_list[1:]
        self.train = flattened_list[:-1]
        # self.final_lyrics = tf.data.Dataset.from_tensor_slices((train, target))
        # self.final_lyrics = docs#np.array(tf.keras.preprocessing.text.text_to_word_sequence(lyrics))

    def build(self):
        _sequences = tf.data.Dataset.from_tensor_slices((self.train, self.target))
        # next(sequences)
        # print(sequences)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((i.numpy() for i,_ in _sequences),target_vocab_size=self.VOCAB_SIZE)
        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus((i.numpy() for _,i in _sequences),target_vocab_size=self.VOCAB_SIZE)

        sample_string = 'Transformer is awesome.'

        tokenized_string = self.tokenizer_en.encode(sample_string)
        print ('Tokenized string is {}'.format(tokenized_string))

        original_string = self.tokenizer_en.decode(tokenized_string)
        print ('The original string: {}'.format(original_string))

        assert original_string == sample_string
        for ts in tokenized_string:
          print ('{} ----> {}'.format(ts, self.tokenizer_en.decode([ts])))

        BUFFER_SIZE = 20000
        BATCH_SIZE = 64

        def encode(lang1, lang2):
            lang1 = self.tokenizer_pt.encode(lang1.numpy())
            lang2 = self.tokenizer_en.encode(lang2.numpy())
            return lang1, lang2

        def tf_encode(pt, en):
            return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

        def filter_max_length(x, y):
            return tf.logical_and(tf.size(x) <= 40,
                                  tf.size(y) <= 40)

        train_dataset = _sequences.map(tf_encode)
        train_dataset = train_dataset.filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        train_dataset = train_dataset.cache()

        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE).skip(TAKE_SIZE).padded_batch(self.BATCH_SIZE, padded_shapes=([40], [40]), drop_remainder=True)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = test_dataset.shuffle(self.BUFFER_SIZE).take(TAKE_SIZE).padded_batch(self.BATCH_SIZE, padded_shapes=([40], [40]), drop_remainder=True)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, test_dataset
