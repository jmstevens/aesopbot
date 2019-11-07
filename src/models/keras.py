import os
import tensorflow
import tensorflow as tf
from tqdm import tqdm
import time
from features.build import Lyrics
import numpy as np

# NOTE: This just a sandbox script, to be treated as a snabox while learning
# tensorflow 2.0. Need to seperate the Lyrics builder class after understadning
# the tf.Data api

# Length of the vocabulary in chars
VOCAB_SIZE = 2**12

# The embedding dimension
embedding_dim = 50

# Number of RNN units
rnn_units = 256

# Batch size
BATCH_SIZE = 64

EPOCHS = 50

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, dropout):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None], mask_zero=True),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(rnn_units,
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

lyrics = Lyrics(BATCH_SIZE, VOCAB_SIZE)
train_dataset, test_dataset = lyrics.build()

for input_example_batch, target_example_batch in train_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()
optimizer = tf.keras.optimizers.Adam(lr=0.005)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                       save_weights_only=True)



def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = lyrics.tokenizer_en.encode(start_string)
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        try:
            text_generated.append(lyrics.tokenizer_tf.decode([predicted_id]))
        except ValueError:
            pass


    return (start_string + ''.join(text_generated))


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

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    loss_avg = []
    for (batch_n, (inp, target)) in enumerate(train_dataset):
        loss = train_step(inp, target)
        loss_avg.append(loss)
        if batch_n % 1000 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))
            print('Epoch {} Average Loss {:.4f}'.format(epoch+1, np.asarray(loss_avg).mean()))
            loss_avg = []
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))
        print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        # # # print('Epoch {} Acc {:.4f}'.format(epoch+1, acc))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        model_test = build_model(vocab_size=VOCAB_SIZE,
                                 embedding_dim=embedding_dim,
                                 rnn_units=rnn_units,
                                 batch_size=1)
        model_test.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        model_test.build(tf.TensorShape([1, None]))
        print(generate_text(model_test, start_string="im only 19 but my mind is older"))
        del model_test

    # print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    # print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    #
    # model.save_weights(checkpoint_prefix.format(epoch=epoch))


# print(generate_text(model, start_string=u"im only 19 but my mind is older"))
