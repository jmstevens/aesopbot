import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import pronouncing
from tqdm import tqdm
import re


class RNNModel:

    def __init__(self,
                 sess,
                 vocabulary,
                 batch_size,
                 sequence_length,
                 hidden_layer_size,
                 cells_size,
                 keep_prob=0.6,
                 gradient_clip=5.,
                 starter_learning_rate=0.01,
                 use_peepholes = False,
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
        self.use_peepholes = use_peepholes
        self.training = training

        # Build Model
        self.build(training)

    def build(self, training):
        if not training:
            self.batch_size = 1
            self.sequence_length = 1

        # Build LSTM
        cells = [rnn.LSTMCell(self.hidden_layer_size, use_peepholes=self.use_peepholes) for _ in range(self.cells_size)]
        cell_drop = [rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob) for cell in cells]
        self.cell = rnn.MultiRNNCell(cell_drop)

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

        # Calculate logits
        logits = tf.matmul(lstm_outputs, self.weights) + self.bias
        self.probabilities = tf.nn.softmax(logits)

        # Train
        total_loss = seq2seq.sequence_loss_by_example([logits],
                                                      [tf.reshape(self.targets, [-1])],
                                                      [tf.ones([self.batch_size * self.sequence_length])],
                                                      self.vocabulary_size)

        self.loss = tf.reduce_sum(total_loss) / self.batch_size / self.sequence_length

        correct_prediction=tf.contrib.metrics.accuracy(tf.argmax([self.probabilities],1),tf.argmax([tf.reshape(self.targets, [-1])],1))
        self.accuracy=tf.cast(correct_prediction,tf.float32)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                           1000, 0.96, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=self.global_step,
                                                name='train_op')

        tf.summary.histogram("logits", logits)
        tf.summary.histogram("probabilities", self.probabilities)
        tf.summary.histogram("loss", self.loss)
        tf.summary.histogram("weights", self.weights)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy", self.accuracy)

    def generate(self, num_out=50, priming_text=None, sample=True):
        # if no priming text is supplied, get a random word to start off the
        # magic.
        state = self.sess.run(self.cell.zero_state(1, tf.float32))
        if priming_text is None:
            prime = np.random.choice(self.vocabulary)
        else:
            prime = priming_text

        # Prime the model]
        word = prime

        try:
            lastword_i = self.vocabulary[word]

        except KeyError:
            # print(self.vocabulary.keys())
            # print(list(self.vocabulary.keys()))
            word = np.random.choice(list(self.vocabulary.keys()))
            # print(word)
            lastword_i = self.vocabulary[word]

        input_i = np.array([[lastword_i]])

        feed_dict = {self.input_data: input_i, self.initial_state: state}
        state = self.sess.run(self.final_state, feed_dict=feed_dict)

        # Generate the text
        gen_seq = prime
        # lastword_rhymes = pronouncing.rhymes(prime)
        counter = 0
        for i in tqdm(range(1, num_out + 1)):

            # generate word probabilities
            input_i = np.array([[lastword_i]])
            feed_dict = {self.input_data: input_i, self.initial_state: state}
            probs, state = self.sess.run([self.probabilities, self.final_state], feed_dict=feed_dict)
            probs = probs[0]

            probs /= probs.sum()
            # print(np.arange(len(probs)))
            # select index of new word
            if sample:
                gen_word_i = np.random.choice(np.arange(len(probs)), p=probs)
            else:
                gen_word_i = np.argmax(probs)

            # append new word
            gen_word = self.vocabulary_inverse[gen_word_i]

            gen_seq += ' ' + gen_word

            lastword_i = gen_word_i

            # probs /= probs.sum()

            # if counter == 24 and lastword_rhymes:
            #     rhymes_list_i = [self.vocabulary[i] for i in lastword_rhymes if i in list(self.vocabulary.keys())]
            #     print("List of rhymes in the vocabulary {}".format(rhymes_list_i))
            #     if rhymes_list_i:
            #         # probs_up = {}
            #         probs_up = np.take(probs, rhymes_list_i)
            #         gen_word_i = rhymes_list_i[np.argmax(np.random.choice(probs_up, 3))]
            #         # gen_word_i = rhymes_list_i[np.argmax(probs_up)]
            #
            #         # gen_seq = gen_seq + '\n'
            #         lastword_i_bar = lastword_i
            #         lastword_rhymes = pronouncing.rhymes(gen_word)
            #     else:
            #         gen_word_i = np.argmax(probs)
            #         # gen_word_i = np.random.choice(np.arange(3, p=probs.argsort()[-3:][::-1]))
            #
            #     gen_word = self.vocabulary_inverse[gen_word_i]
            #     print("generated word {}".format(gen_word))
            #     gen_seq += ' ' + gen_word
            #     lastword_i = gen_word_i
            #
            #     # gen_seq = gen_seq + '\n'
            #     lastword_i_bar = lastword_i
            #     lastword_rhymes = pronouncing.rhymes(gen_word)
            #
            # else:
            #     gen_word_i = np.argmax(probs)
            #     gen_word = self.vocabulary_inverse[gen_word_i]
            #     gen_seq += ' ' + gen_word
            #     lastword_i = gen_word_i
            #
            # if counter == 12:
            #     print(gen_word.split('\n'))
            #     gen_word = gen_word.split('\n')[-1]
            #     gen_word = re.sub('[\W_]','',gen_word)
            #     print("generated word prior {}".format(gen_word))
            #     lastword_rhymes = pronouncing.rhymes(gen_word)
            #     print("last word rhymes prior {}".format(lastword_rhymes))
            #     last_word_to_rhyme = gen_word
            #     # gen_seq = gen_seq + '\n'
            #
            # if counter == 24:
            #     counter = 0

        return ' \n'.join(gen_seq.split('\n'))

    @staticmethod
    def break_apart(sep, step, f):
        return sep.join(f[n:n + step] for n in range(0, len(f), step))
