import tensorflow as tf
from make_dataset import DataProvider
from rnn_model import RNNModel
import sys
import matplotlib
import numpy as np
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

MODELSTATEPATH = os.path.join("../data",os.path.dirname(__file__))
MODELFILE = os.path.join(MODELSTATEPATH, "aesop.ckpt")
BATCH_SIZE = 1
SEQUENCE_LENGTH = 25
LEARNING_RATE = 0.001
DECAY_RATE = 0.9999
HIDDEN_LAYER_SIZE = 256
CELLS_SIZE = 2

TEXT_SAMPLE_LENGTH = 1000
SAMPLING_FREQUENCY = 1000
LOGGING_FREQUENCY = 1000
def generate_text():
    data_provider = DataProvider(BATCH_SIZE, SEQUENCE_LENGTH)
    model = RNNModel(vocabulary_size=data_provider.vocabulary_size, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, hidden_layer_size=HIDDEN_LAYER_SIZE, cells_size=CELLS_SIZE, training=False)
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('../data/aesop.ckpt.meta')
        saver.restore(sess, "../data/aesop.ckpt")
        text = model.sample(sess, data_provider.chars, data_provider.vocabulary, TEXT_SAMPLE_LENGTH).encode("utf-8")
    return text
generate_text(32)
