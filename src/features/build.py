import tensorflow as tf
# import tensorflow_text as text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import json
#import tensorflow_datasets as tfds
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
        self.lyrics = lyrics
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

    def build_char_dataset(self):
        text = ' '.join(self.lyrics) 
        vocab = sorted(set(' '.join(self.lyrics)))
        print(f'Vocab length is {len(vocab)}')

        char2idx = {u:i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        seq_length = 100
        examples_per_epoch = len(text)//(seq_length+1)

        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        for i in char_dataset.take(5):
            print(idx2char[i.numpy()])

        
        sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

        for item in sequences.take(5):
            print(repr(''.join(idx2char[item.numpy()])))

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

        for item in sequences.take(10):
            print(repr(''.join(idx2char[item.numpy()])))

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        for input_example, target_example in  dataset.take(1):
            print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
            print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
        
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        return dataset

if __name__ == '__main__':
    _l = Lyrics(32, 100)
    _l.build_char_dataset()
