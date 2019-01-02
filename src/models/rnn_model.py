import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import pronouncing
from tqdm import tqdm
import re
from math import log


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
        cells = [rnn.GRUCell(self.hidden_layer_size, name='Layer_{}'.format(i)) for i in range(self.cells_size)]
        cell_attention = [rnn.AttentionCellWrapper(cell, attn_length=self.sequence_length) for cell in cells]
        cell_drop = [rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob) for cell in cells]
        self.cell = rnn.MultiRNNCell(cell_drop)
        # self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)

        # Data
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        # self.initial_state = tf.placeholder(tf.float32, [None, self.hidden_layer_size*self.cells_size], name='Hin')

        # Variables
        with tf.variable_scope('lstm_variables', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable('weights', [self.hidden_layer_size, self.vocabulary_size])
            self.bias = tf.get_variable('bias', [self.vocabulary_size])

            # with tf.device('/cpu:0'):
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
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 1000, self.decay_rate, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), self.gradient_clip)
        self.train_op = self.optimizer.apply_gradients(zip(grads, trainable_vars), global_step=self.global_step, name='train_op')#.minimize(self.loss, global_step=self.global_step, name='train_op')#.apply_gradients(zip(grads, trainable_vars), global_step=self.global_step, name='train_op')

        tf.summary.histogram("logits", logits)
        tf.summary.histogram("probabilities", self.probabilities)
        tf.summary.histogram("loss", self.loss)
        tf.summary.histogram("weights", self.weights)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy", self.accuracy)

    @staticmethod
    def beam_search_decoder(data, k):
        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        print(data)
        print(data.tolist())
        for row in data.tolist():
            all_candidates = list()
            print(row)
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -log(row[j])]
                    all_candidates.append(candidate)
                    # order all candidates by score
                    ordered = sorted(all_candidates, key=lambda tup:tup[1])
                    # select k best
                    sequences = ordered[:k]
        return sequences

    def generate(self, num_out=50, priming_text=None, sample=True):
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
            word = np.random.choice(list(self.vocabulary.keys()))
            lastword_i = self.vocabulary[word]
        seq_i = np.array([[lastword_i]])
        input_i = np.array([[lastword_i]])
        feed_dict = {self.input_data: input_i, self.initial_state: state}
        state = self.sess.run(self.final_state, feed_dict=feed_dict)

        # Generate the text
        gen_seq = prime
        counter = 0
        for i in tqdm(range(1, num_out + 1)):
            # generate word probabilities
            # seq_i = np.append(seq_i,input_i)
            input_i = np.array([[lastword_i]])
            # input_i = np.array(seq_i.append(lastword_i))
            # print(seq_i)
            # self.sequence_length = seq_i.shape
            probs, state = self.sess.run([self.probabilities, self.final_state], feed_dict=feed_dict)
            feed_dict = {self.input_data: input_i, self.initial_state: state}
            probs = probs[0]
            # probs = self.beam_search_decoder(probs, 5)
            # print(probs)
            total_sum = np.cumsum(probs)
            sum = np.sum(probs)
            gen_word_i = int(np.searchsorted(total_sum, np.random.rand(1) * sum))
            gen_word = tuple(self.vocabulary.keys())[gen_word_i]

            gen_seq += ' ' + gen_word

            lastword_i = gen_word_i
            # input_i = np.array(seq_i.append(gen_word_i))

        gen_seq = gen_seq.replace('<eol>','\n').replace('<eov','\n').lstrip()
        return gen_seq
