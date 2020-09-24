import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, nn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import pronouncing
from tqdm import tqdm
import re
from math import log
import random
from random import randint
import textwrap

class RNNModel:

    def __init__(self,
                 sess,
                 vocabulary,
                 batch_size,
                 sequence_length,
                 hidden_layer_size,
                 cells_size,
                 keep_prob,
                 gradient_clip,
                 starter_learning_rate,
                 decay_rate,
                 training=True):

        self.sess = sess
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.vocabulary_inverse = {v: k for k, v in self.vocabulary.items()}
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_size = hidden_layer_size
        self.cells_size = cells_size
        self.keep_prob = keep_prob
        self.gradient_clip = gradient_clip
        self.starter_learning_rate = starter_learning_rate
        self.decay_rate = decay_rate
        self.training = training

        # Build Model
        self.build(training)

    def build(self, training):
        if not training:
            self.batch_size = 1
            self.sequence_length = 1


        # Build LSTM
        cells = [rnn.LSTMCell(self.hidden_layer_size, state_is_tuple=True, use_peepholes=False, cell_clip=1., activation='relu') for _ in range(self.cells_size)]
        cell_attention = [rnn.AttentionCellWrapper(cell, attn_length=250, state_is_tuple=True) for cell in cells]
        cell_drop = [rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob) for cell in cell_attention]
        cell = rnn.MultiRNNCell(cell_drop, state_is_tuple=True)
        self.cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Data
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        # Variables
        with tf.variable_scope('lstm_variables', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable('weights', [self.hidden_layer_size, self.vocabulary_size])
            self.bias = tf.get_variable('bias', [self.vocabulary_size])

            with tf.device('/cpu:0'):
                self.embeddings = tf.get_variable('embeddings', [self.vocabulary_size, self.hidden_layer_size])
                # Get embeddings for every input word
                input_embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_data)
                # self.input_embeddings = input_embeddings
                inputs_split = tf.split(input_embeddings, self.sequence_length, 1)
                inputs_split = [tf.squeeze(input_, [1]) for input_ in inputs_split]

        def loop(prev, _):
            previous = tf.matmul(prev, self.ws) + self.bs
            previous_symbol = tf.stop_gradient(tf.argmax(previous, 1))
            return tf.nn.embedding_lookup(self.embeddings, previous_symbol)


        # decoder
        lstm_outputs_split, self.final_state = seq2seq.rnn_decoder(inputs_split,
                                                                   self.initial_state,
                                                                   self.cell,
                                                                   loop_function=loop if not training else None,
                                                                   scope='lstm_variables')

        lstm_outputs = tf.reshape(tf.concat(lstm_outputs_split, 1), [-1, self.hidden_layer_size])

        self.logits = tf.matmul(lstm_outputs, self.weights) + self.bias
        self.probabilities = tf.nn.softmax(self.logits)


        # Train
        total_loss = seq2seq.sequence_loss_by_example([self.logits],
                                                      [tf.reshape(self.targets, [-1])],
                                                      [tf.ones([self.batch_size * self.sequence_length])],
                                                      self.vocabulary_size)

        self.loss = tf.reduce_sum(total_loss) / self.batch_size / self.sequence_length

        correct_prediction=tf.contrib.metrics.accuracy(tf.argmax([self.probabilities],1),tf.argmax([tf.reshape(self.targets, [-1])],1))
        self.accuracy=tf.cast(correct_prediction,tf.float32)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                           100, 0.999, staircase=True)
        trainable_vars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss, trainable_vars)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        tf.summary.histogram("logits", self.logits)
        tf.summary.histogram("probabilities", self.probabilities)
        tf.summary.histogram("loss", self.loss)
        tf.summary.histogram("weights", self.weights)
        # tf.summary.histogram("gradients", gvs)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy", self.accuracy)

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self, provider, num_out=250, priming_text=None, sample=True, temperature=1):
        state = self.sess.run(self.cell.zero_state(1, tf.float32))
        # num_out = randint(50, num_out)
        if priming_text is None:
            # prime = np.random.choice(self.vocabulary)
            lyrics = provider.lyrics
            prime = random.choice(lyrics)[:25]
        else:
            prime = priming_text
        for word in list(prime):
            # print(word)
            try:
                last_word_i = self.vocabulary[word]
            except KeyError:
                _ , last_word_i = random.choice(list(self.vocabulary.items()))
                # print(last_word_i)
                # print(self.vocabulary)
            input_i = np.array([[last_word_i]])

            feed_dict = {self.input_data: input_i, self.initial_state: state}
            state = self.sess.run(self.final_state, feed_dict=feed_dict)

        # generate the sequence

        gen_seq = prime
        input_i = np.array([[last_word_i]])
        # print(input_i)
        for i in range(num_out):
            # generate word probabilities
            # input_i = np.array(word_i)
            input_i = np.array([[last_word_i]])
            # print(input_i)
            feed_dict = {self.input_data: input_i, self.initial_state: state}
            probs, state = self.sess.run([self.probabilities, self.final_state], feed_dict=feed_dict)
            probs = probs[0]
            # print(probs)

            # select index of new word
            if sample:
                gen_word_i = self.sample(probs, temperature)
                # gen_word_i = np.random.choice(np.arange(len(probs)), p=probs)
            else:
                gen_word_i = np.argmax(probs)

            # append new word to the generated sequence
            # print(gen_word_i)
            gen_word = self.vocabulary_inverse[gen_word_i]
            # if gen_word == ' ' and last_word_i == 0:
            #     gen_word = '|'
            # print(gen_word)
            gen_seq += '' + gen_word
            # word_i.append(gen_word_i)
            # print(np.array([word_i]))
            # last_word_i.append(gen_word_i)


            last_word_i = gen_word_i
            # input_i = np.array(word_i) #TODO: use dictionary?

        # gen_seq = gen_seq.replace(" <eol> ",'\n').replace("<eov> ",'').lstrip().replace("<eol> ", '')
        # gen_seq = ' '.join(' '.join(gen_seq.replace(' ','').split('|')).split(' '))
        # print(word_i)
        return gen_seq.replace(' \n ','\n')
