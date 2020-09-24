#!/usr/bin/env python
# coding: utf-8

# In[1]:


## This is code from Google's excellent tutorial on language transformers. I'm applying it to the aesopbot project as a
# learning experiance on language models with attention.
# This is really a the learn language transformers the hard way method of learning notebook
# You can find the tutorial here https://www.tensorflow.org/tutorials/text/transformer
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import json
import os
import time
#:os.chdir('../')
import pickle
import numpy as np
import string, os
from gensim.models import KeyedVectors
import gensim.downloader as api
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from sklearn.model_selection import train_test_split
import random
import sys
from datetime import date
from collections import Counter
import matplotlib.pyplot as plt
from src.features.build import Lyrics
from src.features.transform_data import Transform
from random import shuffle
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# In[2]:


# arr = _t.verse_lines
# verse = arr[20]
# verse
# chunks = [verse[x:x+5] for x in range(0, len(verse), 5)]
# # clean_verse = ' '.join(verse).split()
# # clean_verse
# len(chunks)
# chunk_number = random.randint(1, 10)
# chunks = [verse[x:x+chunk_number] for x in range(0, len(verse), chunk_number)]
# # len([y for x in chunks for y in x]),len(verse)

# max([len(i.split()) for i in verse])


# In[ ]:


def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

def verse_pairs_approach(target_vocab_size=2**12):
    _t = Transform()
    arr = _t.verse_lines
    dataset = list()
    for i in arr:
        tmp = [' \n '.join([clean_text(j[0]), clean_text(j[1])]) for j in zip(i[0::2],i[1::2])]
        dataset.append([z for z in zip(tmp[0::2], tmp[1::2])])
    example = [y[0] for x in dataset for y in x]
    target = [y[1] for x in dataset for y in x]
    X_train, X_test, y_train, y_test = train_test_split(example, target, test_size=0.10, shuffle=True)
    train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)
    return train_examples, val_examples, tokenizer_en, tokenizer_pt

def verse_by_verse(test_size=.10, shuffle=False, target_vocab_size=2**12):
    _t = Transform()
    arr = _t.verse_lines
    dataset = list()
    for verse in arr:
        x = verse[0::2]
        y = verse[1::2]
        [print(i) for i in zip(x, y)]
#         dataset +=
    print(dataset[0])
    if shuffle:
        np.random.shuffle(dataset)
    train = dataset[:round(len(dataset) * test_size)]
    test = dataset[round(len(dataset) * test_size):]

    train_examples = tf.data.Dataset.from_tensor_slices(train)
    val_examples = tf.data.Dataset.from_tensor_slices(test)
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size)
    return train_examples, val_examples, tokenizer_en, tokenizer_pt

# def fill_in_the_blank(test_size=.10, shuffle=False, target_vocab_size=2**12):
#     _t = Transform()
#     arr = _t.verse_lines
#
#     dataset = list()
#     for verse in arr:
#         num_times = random.randint(1, 100)
#         try:
# #             print(max([len(i.split()) for i in verse]))
#             if max([len(i.split()) for i in verse]) > 1 and max([len(i.split()) for i in verse]) < 25:
#                 chunk_number = len(verse) // 5
#                 chunks = [verse[x:x+chunk_number] for x in range(0, len(verse), chunk_number)]
#                 chunk_list = [' '.join(chunk_verse).split() for chunk_verse in chunks]
#                 for chunk in chunk_list:
#                     for i in range(0, num_times,1):
#                         mask = np.random.random(len(chunk))
#                         mask_bool = random.uniform(.1, .9)
#                         mask_x = mask > mask_bool
#                         mask_y = mask < mask_bool
#                         x = ' '.join(['<UNK>' if not x else chunk[i] for i, x in enumerate(mask_x)])
#                         #x = ' '.join(np.array(verse)[mask_x].tolist())
#                         #y = ' '.join(np.array(chunk).tolist())
#                         #$y = ' '.join(['' if not x else chunk[i] for i, x in enumerate(mask_y)])
#                         #y = '|<GAP>|'.join(['' if not x else chunk[i] for i, x in enumerate(mask_y)])
#                         y = ' '.join([np.array(i)[mask_y] for i in chunk])
#                         #y = ' '.join(np.array(verse)[mask_y].tolist())
#                         dataset.append((x, y))
#             else:
#                 pass
#         except ValueError:
#             pass
#     print(dataset[0])
#     example = [x[0] for x in dataset]
#     target = [x[1] for x in dataset]
#     print(len(dataset))
#     X_train, X_test, y_train, y_test = train_test_split(example, target, test_size=0.10, shuffle=True)
#
#     train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#     val_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#     tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#         (pt.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size, reserved_tokens=['<UNK>'])
#
#     tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#         (en.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size) #reserved_tokens=['<UNK>'])
#
#     return train_examples, val_examples, tokenizer_en, tokenizer_pt
#

def fill_in_the_blank(test_size=.10, shuffle=False, target_vocab_size=2**12):
    _t = Transform()
    arr = _t.verse_lines

    dataset = list()
    for verse in arr:
        num_times = random.randint(1, 50)
        try:
            if max([len(i.split()) for i in verse]) > 1 and max([len(i.split()) for i in verse]) < 25:
                chunk_number = len(verse) // 5
                chunks = [verse[x:x+chunk_number] for x in range(0, len(verse), chunk_number)]
                #chunks = ['<START> ' + ''.join([ j for j in verse[x:x+chunk_number]]) for x in range(0, len(verse), chunk_number)]
                #chunks = [chunk for chunk in chunks if len(chunk.split('<NEWLINE>')) > 2]
                chunk_list = [' '.join(chunk_verse).split() for chunk_verse in chunks]

                for chunk in chunk_list:
                    for _ in range(0, num_times,1):
                        mask = np.random.random(len(chunk))
                        mask_bool = random.uniform(.1, .8)
                        mask_x = mask > mask_bool
                        mask_y = mask < mask_bool
                        x = ' '.join(['<UNK>' if not x else chunk[i] for i, x in enumerate(mask_x)])
                        #x = ' '.join(np.array(verse)[mask_x].tolist())
                        #y = ' '.join(np.array(chunk).tolist())
                        y = ' '.join(['' if not x else chunk[i] for i, x in enumerate(mask_y)])
                        #y = '|<GAP>|'.join(['' if not x else chunk[i] for i, x in enumerate(mask_y)])
                        #y = ' '.join(['<UNK>' if x else chunk[i] for i, x in enumerate(mask_x)])
                         # = ' '.join([np.array(i)[mask_y] for i in chunk])
                        # x = ' '.join(np.array(chunk)[mask_x].tolist())
                        #y = ' '.join(np.array(chunk)[mask_y].tolist())
                        #x = ' '.join([' ' if not x else chunk.split(' ')[i] for i, x in enumerate(mask_x)])
                        #x = ' '.join([' ' if not x else chunk.split(' ')[i] for i, x in enumerate(mask_x)])
                        y = ' '.join(chunk)
                        dataset.append((x, y))
        except ValueError:
            pass
    print(dataset[0])
    example = [x[0] for x in dataset]
    target = [x[1] for x in dataset]
    print(len(dataset))
    X_train, X_test, y_train, y_test = train_test_split(example, target, test_size=0.10, shuffle=True)

    train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size, reserved_tokens=['<UNK>'])

    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=target_vocab_size,reserved_tokens=['<UNK>'])

    BUFFER_SIZE = 5000
    BATCH_SIZE = 32

    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size+1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size+1]
        return lang1, lang2

    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    MAX_LENGTH = 125



    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                            tf.size(y) <= max_length)


    #train_dataset = train_examples.map(tf_encode)
    #train_dataset = train_dataset.filter(filter_max_length)
    ## cache the dataset to memory to get a speedup while reading from it.
    #train_dataset = train_dataset.cache()
#   # train_dataset = train_dataset.repeat(25)
    #train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


    #val_dataset = val_examples.map(tf_encode)
    #val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

    return train_examples, val_examples, tokenizer_en, tokenizer_pt

train_examples, val_examples, tokenizer_en, tokenizer_pt = fill_in_the_blank(target_vocab_size=2**13)


# In[ ]:


duration = 5  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


# In[ ]:


# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
#                                as_supervised=True)
# train_examples, val_examples = examples['train'], examples['validation']
# train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# val_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (en.numpy() for pt, en in train_examples), target_vocab_size=2**12)

# tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (pt.numpy() for pt, en in train_examples), target_vocab_size=2**12)


# In[ ]:


sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string


# In[ ]:


for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


# In[ ]:


BUFFER_SIZE = 5000
BATCH_SIZE = 64


# In[ ]:


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2


# In[ ]:


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


# In[ ]:


MAX_LENGTH = 100


# In[ ]:


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)


# In[ ]:


train_preprocessed = (
        train_examples
        .map(tf_encode)
        .filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        .cache()
        .shuffle(BUFFER_SIZE))

val_preprocessed = (
 val_examples
 .map(tf_encode)
 .filter(filter_max_length))
#train_dataset = train_examples.map(tf_encode)
#train_dataset = train_dataset.filter(filter_max_length)
# # cache the dataset to memory to get a speedup while reading from it.
#train_dataset = train_dataset.cache()
#train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
#train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
#
#val_dataset = val_examples.map(tf_encode)
#val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)


# In[ ]:





# In[ ]:


train_dataset = (train_preprocessed
                  .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
                  .prefetch(tf.data.experimental.AUTOTUNE))


val_dataset = (val_preprocessed
                .padded_batch(BATCH_SIZE,  padded_shapes=([None], [None])))


# In[ ]:


pt_batch, en_batch = next(iter(val_dataset))
pt_batch, en_batch


# In[ ]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


# In[ ]:


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# In[ ]:


pos_encoding = positional_encoding(50, 512)
print (pos_encoding.shape)

#plt.pcolormesh(pos_encoding[0], cmap='RdBu')
#plt.xlabel('Depth')
#plt.xlim((0, 512))
#plt.ylabel('Position')
#plt.colorbar()
#plt.show()


# In[ ]:
tokenizer_en.encode

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_unk_mask(seq):
    seq = tf.cast(tf.math.equal(seq, tokenizer_en.encode('<UNK>')[0]), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# In[ ]:


x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)


# In[ ]:


# The look-ahead mask is used to mask the future tokens in a sequence.
# In other words, the mask indicates which entries should not be used.

#This means that to predict the third word, only the first and second word will be used. Similarly to predict the fourth word,
#only the first, second and the third word will be used and so on.

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)
x = tf.random.uniform((0,3))
temp = create_look_ahead_mask(x.shape[1])
temp


# In[ ]:


def scaled_dot_product_attention(q, k, v, mask):
    """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
          but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights


    """
    matmul_qk = tf.matmul(q, k, transpose_b=True) #(..., seq_len, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) for scores to add up to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights
#As the softmax normalization is done on K, its values decide the amount of importance given to Q.


# In[ ]:


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None
    )
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


# In[ ]:


np.set_printoptions(suppress=True)

temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32) # (4, 3)
temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32) # (4, 2)
# this query aligns with the second key
# so the second value is returned
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32) # (1, 3)
print_out(temp_q, temp_k, temp_v)


# In[ ]:


# This query aligns with a repeated key (third and fourth)
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
print_out(temp_q, temp_k, temp_v)


# In[ ]:


temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32) # (3, 3)
print_out(temp_q, temp_k, temp_v)


# In[ ]:


# Multiheaded attention
# Instead of one single attention head, Q, K, and V are split into multiple heads
# because it allows the model to jointly attend to information at different positions from
# different representational spaces. After the split each head has a reduced dimensionality,
# so the total computation cost is the same as a single head attention with full dimensionality.
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
           Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wq(k) # (batch_size, seq_len, d_model)
        v = self.wq(v) # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


# In[ ]:


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512)) # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
out.shape, attn.shape


# In[ ]:


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),# (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
    ])


# In[ ]:


sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape


# In[ ]:


# Encoder Layer
#Each encoder layer consists of sublayers:
#    Multi-head attention (with padding mask)
#    Point wise feed forward networks.
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)

        return out2




# In[ ]:


sample_encoder_layer = EncoderLayer(512, 8, 2048)

sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
sample_encoder_layer_output.shape # (batch_size, input_seq_len, d_model)


# In[ ]:


# Each of these sublayers has a residual connection around it followed by a layer normalization.
# The output of each sublayer is LayerNorm(x + Sublayer(x)). The normalization is done on the d_model (last) axis.

# There are N decoder layers in the transformer.

# As Q receives the output from decoder's first attention block, and K receives the encoder output,
# the attention weights represent the importance given to the decoder's input based on the encoder's output.
# In other words, the decoder predicts the next word by looking at the encoder output and self-attending to its
# own output. See the demonstration above in the scaled dot product attention section.

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, input_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x) # (batch_size, input_seq_len, d_model)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask) # (batch_size, input_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1) # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out2) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out3 = self.layernorm2(ffn_output + out2) # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2



# In[ ]:


sample_decoder_layer = DecoderLayer(512, 8, 2048)

sample_decoder_layer_output, _, _ = sample_decoder_layer(
    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
    False, None, None
)
sample_decoder_layer_output.shape # (batch_size, target_seq_len, d_model)


# In[ ]:


# Encoder consitsts of
    # 1. Input Embedding
    # 2. Positional Encoding
    # 3. N encoder layers
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # embedding and position encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)


        return x # (batch_size, input_seq_len, d_model)



# In[ ]:


sample_encoder = Encoder(num_layers=2, d_model=512,
                         num_heads=8, dff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)

temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

print(sample_encoder_output.shape)


# In[ ]:


# Decoder
# 1.Output Embedding
# 2. Positional Encoding
# 3. N decoder layers

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights



# In[ ]:


sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000, maximum_position_encoding=5000)
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
output, attn = sample_decoder(temp_input, enc_output=sample_encoder_output,
                              training=False, look_ahead_mask=False, padding_mask=False)
output.shape, attn['decoder_layer2_block2'].shape


# In[ ]:


# Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                              input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                              target_vocab_size, pe_target, rate)


        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask) # (batch_size, inp_seq_len, d_model)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights



# In[ ]:


sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048,
    input_vocab_size=8500, target_vocab_size=8000,
    pe_input=10000, pe_target=6000
)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                              enc_padding_mask=None,
                              look_ahead_mask=None,
                              dec_padding_mask=None)

fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)


# In[ ]:


num_layers = 6
d_model = 512
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.35


# In[ ]:


# Optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# In[ ]:


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


# In[ ]:


temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning rate")
plt.xlabel("Train Step")


# In[ ]:


# Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


# In[ ]:


# Define our metrics
#get_ipython().run_line_magic('load_ext', 'tensorboard')
import datetime
#get_ipython().system('rm -rf ./logs/')
#file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')



current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'log/gradient_tape/' + current_time + '/train'
test_log_dir = 'log/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


# In[ ]:


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# In[ ]:


checkpoint_path = "./checkpoints_language/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

#if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#    ckpt.restore(ckpt_manager.latest_checkpoint)
#    print ('Latest checkpoint restored!|!')


# In[ ]:


EPOCHS = 250

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
tf.TensorSpec(shape=(None, None), dtype=tf.int64),
tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

test_step_signature = [
tf.TensorSpec(shape=(None, None), dtype=tf.int64),
tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

@tf.function(input_signature=test_step_signature)
def test_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(inp, tar_inp,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
    loss = loss_function(tar_real, predictions)
    test_loss(loss)
    test_accuracy(tar_real, predictions)


# In[ ]:


def evaluate(inp_sentence, temperature):
    text_generated = []
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]
        # using a categorical distribution to predict the word returned by the model
        predictions = tf.squeeze(predictions, 0)
#         print(tf.math.top_k(predictions, 3))
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        predicted_id = tf.expand_dims([predicted_id], 0)
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_pt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result)-1.5, -0.5)

        ax.set_xticklabels(
            ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                            if i < tokenizer_en.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head+1))

    plt.tight_layout()
    plt.show()

def translate(sentence, plot='', temperature=1.0):
    result, attention_weights = evaluate(sentence, temperature)

    predicted_sentence = tokenizer_en.decode([i for i in result
                                            if i < tokenizer_en.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))
    #print(attention_weights)
    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)
    return tf.constant(predicted_sentence), attention_weights


# In[ ]:


for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    with train_summary_writer.as_default():
        tf.summary.trace_on(
            graph=True, profiler=True
        )
        tf.summary.trace_export(name="train_step",step=batch,profiler_outdir='logs/gradient_tape/')
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        output, attention_weights = translate("bitches on my dick because I look like jesus",plot='',temperature=.5)
        tf.summary.text('generated text', output, step=epoch)
        for i in range(transformer.decoder.num_layers):
            tf.summary.histogram('decoder_layer{}_block1'.format(i+1), attention_weights['decoder_layer{}_block1'.format(i+1)], step=epoch)
            tf.summary.histogram('decoder_layer{}_block2'.format(i+1), attention_weights['decoder_layer{}_block2'.format(i+1)], step=epoch)


    for (batch, (inp, tar)) in enumerate(val_dataset):
        test_step(inp, tar)
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Test Loss {:.4f} Test Accuracy {:.4f}'.format(
                        epoch + 1, batch, test_loss.result(), test_accuracy.result()))
    with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)


    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                    train_loss.result(),
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()



# In[ ]:
