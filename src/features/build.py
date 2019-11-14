import tensorflow as tf
# import tensorflow_text as text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import json
import tensorflow_datasets as tfds
import pickle
from nltk.tokenize import WordPunctTokenizer


class Lyrics:
    BUFFER_SIZE = 10000
    with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)

    def __init__(self, BATCH_SIZE, VOCAB_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.VOCAB_SIZE = VOCAB_SIZE

        data_dir = 'data/processed/verses.txt'
        with open(data_dir, "rb") as fp:   # Unpickling
            lyrics = pickle.load(fp)
        # [print(i) for i in lyrics]
        # lyrics = [' \n '.join(tokenizer.tokenize(i)) for i in lyrics]
        lyrics = np.array(lyrics)
        arr = [[j for j in i.split(' \n ') if len(j) > 1 and '\n\n' != j] for i in list(np.array(lyrics)) if len(i.split(' \n ')) > 0]
        flattened_list = np.asarray([y for x in arr for y in x])
        print(flattened_list)
        self.target = flattened_list[1:]
        self.train = flattened_list[:-1]

    def build(self, pad_shape=40):
        _sequences = tf.data.Dataset.from_tensor_slices((self.train, self.target))

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((i.numpy() for i,_ in _sequences),
                                                                                    target_vocab_size=self.VOCAB_SIZE)
        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus((i.numpy() for _,i in _sequences),
                                                                                    target_vocab_size=self.VOCAB_SIZE)

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
        TAKE_SIZE = 5000


        def encode(lang1, lang2):
            lang1 = self.tokenizer_pt.encode(lang1.numpy())# + [self.tokenizer_pt.vocab_size+1]
            lang2 = self.tokenizer_en.encode(lang2.numpy())# + [self.tokenizer_en.vocab_size+1]
            return lang1, lang2

        def tf_encode(pt, en):
            return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

        def filter_max_length(x, y):
            return tf.logical_and(tf.size(x) <= pad_shape,
                                  tf.size(y) <= pad_shape)

        dataset = _sequences.map(tf_encode)
        dataset = dataset.filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        # dataset = dataset.cache()
        #.shuffle(self.BUFFER_SIZE).cache()
        dataset = dataset.cache().padded_batch(self.BATCH_SIZE,
                                             padded_shapes=([pad_shape],
                                                            [pad_shape]),
                                             drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # test_dataset = dataset.shuffle(self.BUFFER_SIZE).cache().take(TAKE_SIZE)\
        #                       .padded_batch(self.BATCH_SIZE,
        #                                     padded_shapes=([pad_shape],
        #                                                    [pad_shape]),
        #                                     drop_remainder=True)
        # test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
