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
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku 
import random
import sys
# NOTE: This just a sandbox script, to be treated as a snabox while learning
# tensorflow 2.0. Need to seperate the Lyrics builder class after understadning
# the tf.Data api


## TODO: Add tqdm template for loss

# Length of the vocabulary in chars
VOCAB_SIZE = 2**13


# The embedding dimension
embedding_dim = 50

# Number of RNN units

rnn_units = 512


# Batch size
BATCH_SIZE = 64

EPOCHS = 50

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, dropout):
    input_shape = (max_sequence_len,)
    print(max_sequence_len)
    sentence_indices = Input(shape=input_shape, dtype='int32')
        
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_vectors, tokenizer.word_index)

    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices) 

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences.
    X = LSTM(units=128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(units=128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X through a Dense layer with 5 units
    X = Dense(units=total_words)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)



    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    model.summary()
    return model


model = build_model(
            vocab_size=VOCAB_SIZE,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            batch_size=BATCH_SIZE,
            dropout=0.1)

lyrics = Lyrics(BATCH_SIZE, VOCAB_SIZE)

dataset = lyrics.build(pad_shape=40)

# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()
optimizer = tf.keras.optimizers.Adam(lr=0.01)


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                       save_weights_only=True)




def generate_text(model_gen, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 100

    # Converting our start string to numbers (vectorizing)
    input_eval = lyrics.tokenizer_pt.encode(start_string)

    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model_gen.reset_states()
    for i in range(num_generate):
        predictions = model_gen(input_eval)

        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        try:
            text_generated.append(lyrics.tokenizer_en.decode([predicted_id]))

        except ValueError:
            pass

    return (start_string + ' '.join(text_generated))


@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

# Training step


@tf.function
def test_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))

    return loss

# summary_writer = tf.summary.create_file_writer('./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))


for epoch in range(EPOCHS):
    start = time.time()
    # avg_loss = tf.keras.metrics.Mean()
    # avg_test_loss = tf.keras.metrics.Mean()
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    print(f"Epoch {epoch + 1} of {EPOCHS}")
    # with summary_writer.as_default():
    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)
        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))
            # avg_loss.update_state(loss)
    # for (batch_n_test, (inp_test, target_test)) in enumerate(test_dataset):
    #     test_loss = test_step(inp_test, target_test)
    #     if batch_n_test % 100 == 0:
    #         template = 'Epoch {} Batch {} Test Loss {}'
    #         print(template.format(epoch+1, batch_n_test, test_loss))
            # avg_test_loss.update_state(test_loss)
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 1 == 0:

        model.save_weights(checkpoint_prefix.format(epoch=epoch))
        print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        # # # print('Epoch {} Acc {:.4f}'.format(epoch+1, acc))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        # model_test = build_model(vocab_size=VOCAB_SIZE,
        #                          embedding_dim=embedding_dim,
        #                          rnn_units=rnn_units,
        #                          batch_size=1,
        #                          dropout=0.1)
        # model_test.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        # model_test.build(tf.TensorShape([1, None]))
        # print(generate_text(model_test, start_string="im only 19 but my mind is older"))

    # tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
    # tf.summary.scalar('test_loss', avg_test_loss.result(), step=optimizer.iterations)
    # avg_loss.reset_states()
    # avg_test_loss.reset_states()

