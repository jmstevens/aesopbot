import tensorflow
import tensorflow as tf
import json
import os
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
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
today = date.today()
# Load vectors directly from the file
word_vectors = api.load("glove-twitter-100")
num_words = 10000
# with open('configs/config.json','r') as cfgFile:
#     cfg = json.load(cfgFile)
# data_dir = 'data/processed/verses.txt'
#
# with open(data_dir, "rb") as fp:   # Unpickling
#     lyrics = pickle.load(fp)

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt
_t = Transform()
lyrics = _t.verse_lines
print(lyrics[0])
arr = np.array(lyrics)
#arr = [[clean_text(j) for j in i.split(' \n ') if len(j) > 1 and '\n\n' != j] for i in list(np.array(lyrics)) if len(i.split(' \n ')) > 0]
#print(arr[0])

# np.random.shuffle(arr)
#flattened_list = [y for x in lyrics for y in x]
flattened_list = np.asarray([clean_text(y) for x in arr for y in x if len(y.split()) <= 30])
print(flattened_list[0])
tokenizer = Tokenizer()
#corpus = flattened_list
corpus = [' '.join(i) for i in flattened_list]
def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print(f'Number of words before downsampling: {total_words}')
    count_thres = 1
    #low_count_words = [w for w,c in tokenizer.word_counts.items() if c < count_thres]
    #for w in low_count_words:
    #    del tokenizer.word_index[w]
    #    del tokenizer.word_docs[w]
    #    del tokenizer.word_counts[w]
    total_words = len(tokenizer.word_index) + 1
    print(f'Number of words after downsampling: {total_words}')
    # convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words
inp_sequences, total_words = get_sequence_of_tokens(corpus)
num_words = total_words
print(inp_sequences[:10])

input_sequences = inp_sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len+1, padding='pre'))

predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
print(predictors.shape, label.shape)
X_train, X_test, y_train, y_test = train_test_split(predictors, label, test_size=0.10, shuffle=True, random_state=42)
print(word_vectors.get_vector('word'))
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
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
    embedding_layer = tensorflow.keras.layers.Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=True, mask_zero=True)
    ### END CODE HERE ###

    # Step 4 (already done for you; please do not modify)
    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer



embedding = pretrained_embedding_layer(word_vectors, tokenizer.word_index)
input_shape = (max_sequence_len,64)
print(max_sequence_len)
sentence_indices = Input(shape=input_shape, dtype='int32')

# Create the embedding layer pretrained with GloVe Vectors (≈1 line)
embedding_layer = pretrained_embedding_layer(word_vectors, tokenizer.word_index)

# Propagate sentence_indices through your embedding layer
# (See additional hints in the instructions).
embeddings = embedding_layer(sentence_indices)

# Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
# The returned output should be a batch of sequences.
X = LSTM(units=2048, return_sequences=True, batch_size=64, stateful=True)(embeddings)
#X = BatchNormalization(axis=-1)(X)
# Add dropout with a probability of 0.5
X = Dropout(rate=0.5)(X)
# Propagate X trough another LSTM layer with 128-dimensional hidden state
# The returned output should be a single hidden state, not a batch of sequences.
X = LSTM(units=2048, return_sequences=False)(X)
# Add dropout with a probability of 0.5
#X = BatchNormalization(axis=-1)(X)
X = Dropout(rate=0.5)(X)
# Propagate X through a Dense layer with 5 units
X = Dense(units=num_words)(X)
# Add a softmax activation
X = Activation('softmax')(X)



# Create Model instance which converts sentence_indices into X.
model = Model(inputs=sentence_indices, outputs=X)
model.summary()
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    text = flattened_list
    start_index = random.randint(0, len(text) - max_sequence_len - 1)
    for diversity in [0.2, 0.5, .75, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(100):
            #x_pred = [tokenizer.word_index[i] for i in sentence.split()]
            #x_pred = np.array(pad_sequences([[tokenizer.word_index[i] for i in sentence.split()]], maxlen=max_sequence_len, padding='post'))

            x_pred = np.array(pad_sequences(tokenizer.texts_to_sequences([sentence]), maxlen=max_sequence_len, padding='post'))
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            #try:
            next_char = ' ' + tokenizer.index_word[next_index]
            #except KeyError:
            #    next_char = ''

            sentence = sentence + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

def plot_history(history):
    date = today.strftime("%m_%d_%y")

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'src/visualization/aesop_accuracy_{date}.png')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'src/visualization/aesop_loss_{date}.png')
    return True




print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
history = model.fit(X_train, y_train, epochs = 50, batch_size = 64, validation_split=0.15, shuffle=False, callbacks=[print_callback, checkpointer])
plot_history(history)
date = today.strftime("%m_%d_%y")
model.save(f'data/aesopbot_{date}.hd5')
model.evaluate(X_test, y_test, batch_size = 64)
model.save(f'data/aesopbot_{date}.hd5')
