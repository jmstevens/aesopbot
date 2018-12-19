from tensorflow import keras
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import tensorflow.keras.utils as ku
import numpy as np
import random

tokenizer = Tokenizer()

def dataset_preparation(data):
    corpus = data.lower().split("\n")
    corpus = [text_to_word_sequence(i) for i in corpus if len(i) > 1]
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    output = predictors, label, max_sequence_len, total_words
    return output

def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1

    model = Sequential()
    model.add(Embedding(total_words, 32, input_length=input_len))
    model.add(LSTM(256))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)

def generate_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,
                                   padding='pre')
        predicted = model.predicted_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

def run():
    data = open('data/processed/lyrics.txt').read()
    X, Y, max_seq_len, total_words = dataset_preparation(data)
    model = create_model(X, Y, max_seq_len, total_words)
    return generate_text(random.sample(data.split("\n"), 1), 3, max_sequence_len, model)

if __name__ == "__main__":
    run()
