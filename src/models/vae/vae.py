import tensorflow
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from tensorflow.keras.activations import elu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import gensim.downloader as api
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import csv
import os
from tokenizers import CharBPETokenizer, BertWordPieceTokenizer
import nltk
nltk.download('genesis')
nltk.download('nps_chat')
nltk.download('webtext')
nltk.download('treebank')
from tqdm import tqdm
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from src.features.build import Transform
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

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

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words


def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    txt = normalize(txt.split(" "))
    return txt

def run():
    word_vectors = api.load("glove-twitter-25")
    VALIDATION_SPLIT = 0.2
    MAX_SEQUENCE_LENGTH = 500
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 10
    _t = Transform()
    data = _t.verse_lines
    cleaned_text_array = [[' '.join(clean_text(i)) for i in verse] for verse in tqdm(data)]
    cleaned_verse_w_lines_joined = [' '.join([y for y in x if len(y.split()) <= 30]) for x in cleaned_text_array]
    cleaned_verse_w_lines_joined_array = np.array(cleaned_verse_w_lines_joined)
    tokenizer = Tokenizer(MAX_NB_WORDS+1, oov_token='unk') #+1 for 'unk' token
    tokenizer.fit_on_texts(cleaned_verse_w_lines_joined_array)
    print('Found %s unique tokens' % len(tokenizer.word_index))
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS} # <= because tokenizer is 1 indexed
    tokenizer.word_index[tokenizer.oov_token] = MAX_NB_WORDS + 1
    word_index = tokenizer.word_index #the dict values start from 1 so this is fine with zeropadding
    index2word = {v: k for k, v in word_index.items()}
    sequences = tokenizer.texts_to_sequences(cleaned_verse_w_lines_joined_array)
    sequences = [i for i in sequences if len(i) < 500]
    data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data_1.shape)
    NB_WORDS = (min(tokenizer.num_words, len(word_index))+1) #+1 for zero padding

    data_val = data_1#[len(data_1) - round(len(data_1) * .10):]
    data_train = data_1#[:len(data_1) - round(len(data_1) * .10)]

    def pretrained_embedding_layer(word_to_vec_map, word_to_index):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

        Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """

        vocab_len = len(word_to_index) + 2                  # adding 1 to fit Keras embedding (requirement)
        emb_dim = word_vectors.get_vector("cucumber").shape[0]      # define dimensionality of your GloVe word vectors (= 50)
        emb_dim_one = word_vectors.get_vector("cucumber").shape

        ### START CODE HERE ###
        # Step 1
        # Initialize the embedding matrix as a numpy array of zeros.
        # See instructions above to choose the correct shape.
        emb_matrix = np.zeros((vocab_len, emb_dim))

        # Step 2
        # Set each row "idx" of the embedding matrix to be
        # the word vector representation of the idx'th word of the vocabulary
        for word, idx in word_to_index.items():
            try:
                emb_matrix[idx, :] = word_vectors.get_vector(word)
            except KeyError:
                emb_matrix[idx, :] = np.zeros(word_vectors.get_vector("cucumber").shape)

        # Step 3
        # Define Keras embedding layer with the correct input and output sizes
        # Make it non-trainable.
        embedding_layer = tensorflow.keras.layers.Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False, mask_zero=True)
        ### END CODE HERE ###

        # Step 4 (already done for you; please do not modify)
        # Build the embedding layer, it is required before setting the weights of the embedding layer.
        embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.

        # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
        embedding_layer.set_weights([emb_matrix])

        return embedding_layer

    embedding = pretrained_embedding_layer(word_vectors, tokenizer.word_index)
    batch_size = 32
    max_len = MAX_SEQUENCE_LENGTH
    emb_dim = EMBEDDING_DIM
    latent_dim = 64
    intermediate_dim = 256
    epsilon_std = 1.0
    kl_weight = 0.01
    num_sampled = 500

    x = Input(batch_shape=(None, max_len))
    x_embed = embedding(x)
    h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
    h = Dropout(0.2)(h)
    h = Dense(intermediate_dim, activation='linear')(h)
    h = elu(h)
    h = Dropout(0.2)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,stddev=epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    repeated_context = RepeatVector(max_len)
    decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
    decoder_mean = TimeDistributed(Dense(NB_WORDS, activation='linear'))#softmax is applied in the seq2seqloss by tf
    # decoder_mean = Dense(NB_WORDS, activation='linear')
    h_decoded = decoder_h(repeated_context(z))
    x_decoded_mean = decoder_mean(h_decoded)

    def zero_loss(y_true, y_pred):
        return K.zeros_like(y_pred)

    #
    # class BahdanauAttention(tf.keras.layers.Later):
    #     def __init__(self, units):
    #         super(BahdanauAttention, self).__init__()
    #         self.W1 = tf.keras.layers.Dense(units)
    #         self.W1 = tf.keras.layers.Dense(units)
    #         self.V = tf.keras.layers.Dense(1)
    #
    #     def call(self, query, values):
    #         query_with_time_axis = tf.expand_dims(query, 1)
    #         score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    #
    #         attention_weights = tf.nn.softmax(score, axis=1)
    #
    #         context_vector = attention_weights * values
    #         context_vector = tf.reduce_sum(context_vector, axis=1)
    #
    #         return context_vector, attention_weights

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
            self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

        def vae_loss(self, x, x_decoded_mean):
            print(x.shape, x_decoded_mean.shape)
            #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
            labels = tf.cast(x, tf.int32)
            xent_loss = K.sum(tfa.seq2seq.sequence_loss(x_decoded_mean, labels,
                                                         weights=self.target_weights,
                                                         average_across_timesteps=False,
                                                         average_across_batch=False), axis=-1)#,
                                                         #softmax_loss_function=softmax_loss_f), axis=-1)#,
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            xent_loss = K.mean(xent_loss)
            kl_loss = K.mean(kl_loss)
            return K.mean(xent_loss + kl_weight * kl_loss)


        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            print(x.shape, x_decoded_mean.shape)
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # we don't use this output, but it has to have the correct shape:
            return K.ones_like(x)

    def kl_loss (x, x_decoded_mean):
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = kl_weight * kl_loss

    loss_layer = CustomVariationalLayer()(x_decoded_mean)
    vae = Model(x, [loss_layer])
    opt = Adam(lr=0.01)
    vae.compile(optimizer='adam')

    def create_model_checkpoint(dir, model_name):
        filepath = dir + '/' + model_name + ".h5" #-{epoch:02d}-{decoded_mean:.2f}
        directory = os.path.dirname(filepath)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)
        return checkpointer

    checkpointer = create_model_checkpoint('models', 'vae_seq2seq')

    # nb_epoch=100
    # n_steps = (800000/2)/batch_size #we use the first 800000
    # for counter in range(nb_epoch):
    #     print('-------epoch: ',counter,'--------')
    vae.fit(x=data_train, y=data_train, batch_size=batch_size, epochs=10, callbacks=[checkpointer])

    vae.save('models/vae_lstm800k32dim96hid.h5')
