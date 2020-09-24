import os
import tensorflow
import tensorflow as tf
from tqdm import tqdm
import time
from src.features.build import Lyrics
import numpy as np
import tensorflow_datasets as tfds
import datetime as dt
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# NOTE: This just a sandbox script, to be treated as a snabox while learning
# tensorflow 2.0. Need to seperate the Lyrics builder class after understadning
# the tf.Data api


## TODO: Add tqdm template for loss

# Length of the vocabulary in chars
VOCAB_SIZE = 2**13


# The embedding dimension
embedding_dim = 256

# Number of RNN units

rnn_units = 2048


# Batch size
BATCH_SIZE = 32

EPOCHS = 50
lyrics = Lyrics(BATCH_SIZE, VOCAB_SIZE)

dataset = lyrics.build(pad_shape=10)
#VOCAB_SIZE = len(lyrics.vocab)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, dropout):
    model = tf.keras.Sequential([

        tf.keras.layers.Embedding(vocab_size+1, embedding_dim,
                              batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
         tf.keras.layers.Dropout(dropout),
         tf.keras.layers.GRU(rnn_units,
                              return_sequences=True,
                              stateful=True,
                              recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(vocab_size)
        ])
    return model


model = build_model(
            vocab_size=VOCAB_SIZE,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            batch_size=BATCH_SIZE,
            dropout=0.1)


# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()
optimizer = tf.keras.optimizers.Adam(lr=0.01)

#print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
#  checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
#  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
#  history = model.fit(dataset, verbose=1, epochs = 50, callbacks=[checkpointer])
#  plot_history(history)
#  date = today.strftime("%m_%d_%y")
#  model.save(f'data/aesopbot_{date}.hd5')
#  model.evaluate(X_test, y_test, batch_size = 64)
#  model.save(f'data/aesopbot_{date}.hd5')

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                       save_weights_only=True)




def generate_text(model_gen, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000
    #input_eval = [lyrics.char2idx[s] for s in start_string]
    #input_eval = tf.expand_dims(input_eval, 0)

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
        #predictions = lyrics.tokenizer_en.decode(predictions)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(predicted_id) #lyrics.tokenizer_en.decode(predicted_id))


    return text_generated#(start_string + ''.join(text_generated))


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


#@tf.function
#def test_step(inp, target):
#    with tf.GradientTape() as tape:
#        predictions = model(inp)
#        loss = tf.reduce_mean(
#            tf.keras.losses.sparse_categorical_crossentropy(
#                target, predictions, from_logits=True))
#
#    return loss

summary_writer = tf.summary.create_file_writer('./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))


for epoch in range(EPOCHS):
    start = time.time()
    avg_loss = tf.keras.metrics.Mean()
    # avg_test_loss = tf.keras.metrics.Mean()
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    print(f"Epoch {epoch + 1} of {EPOCHS}")
    # with summary_writer.as_default():
    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)
        if batch_n % 1000 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))
            avg_loss.update_state(loss)
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
        #print('Epoch {} Acc {:.4f}'.format(epoch+1, acc))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        #  model_test = build_model(vocab_size=VOCAB_SIZE,
        #                          embedding_dim=embedding_dim,
        #                          rnn_units=rnn_units,
        #                          batch_size=1,
        #                          dropout=0.1)
        #  model_test.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        #  model_test.build(tf.TensorShape([1, None]))
        #  print(generate_text(model_test, start_string="im only 19 but my mind is older"))

    tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
    # tf.summary.scalar('test_loss', avg_test_loss.result(), step=optimizer.iterations)
    avg_loss.reset_states()
    # avg_test_loss.reset_states()
