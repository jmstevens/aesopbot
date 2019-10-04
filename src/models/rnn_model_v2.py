import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from src.features.build import Lyrics
import os
from time import time


EPOCHS=25
_l = Lyrics(32)
dataset, word2idx, idx2word, vocab_size = _l.build_words()
optimizer = tf.keras.optimizers.Adam(lr=0.001)
checkpoint_dir = './src/data/tensorboard/'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size, embedding):
    model = tf.keras.Sequential([
        tf.keras.Input(input_shape, dtype=tf.int64)
        hub_layer,
        tf.keras.layers.LSTM(rnn_units,return_sequences=True, stateful=True),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.LSTM(rnn_units,return_sequences=True, stateful=True),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.LSTM(rnn_units,return_sequences=True, stateful=True),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True,stateful=True),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(1500, activation='relu'),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(vocab_size, activation='softmax')])
    return model

model = build_model(
              vocab_size = vocab_size,
              embedding_dim=250,
              rnn_units=256,
              batch_size=32
              )

@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=False))
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(
                target, predictions))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, acc

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

def generate_text(model_test, start_string, word2idx):
    input_eval = [word2idx[s] for s in start_string.split(' ') if s != '']
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model_test.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model_test.build(tf.TensorShape([1, None]))
    model_test.reset_states()
    for i in range(250):
        predictions = model_test(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / 1
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2word[predicted_id])
    print((start_string + ' '.join(text_generated)).replace(' eol ', '\n').replace(' eov ', '\n\n'))
    return True


for epoch in range(EPOCHS):
    start = time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss, acc = train_step(inp, target)

        if batch_n % 10 == 0:
            template = 'Epoch {} Batch {} Loss {} Acc {}'
            print(template.format(epoch+1, batch_n, loss, acc))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 1 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))
        # train_acc = acc.result()
        # acc.reset_states()
        # for x_batch_val, y_batch_val in test_data:
        #    val_logits = model(x_batch_val)
        #    val_acc_metric(y_batch_val, val_logits)
        # val_acc = val_acc_metric.result()
        # val_acc_metric.reset_states()
        print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print('Epoch {} Acc {:.4f}'.format(epoch+1, acc))
        print('Time taken for 1 epoch {} sec\n'.format(time() - start))
        # print('Validation acc: %s' % (float(val_acc)))
        start_string = "hiphop saved my life "
        model_test = build_model(vocab_size, 250, 256, batch_size=1)
        generate_text(model_test, start_string, word2idx)
        del model_test

    model.save_weights(checkpoint_prefix.format(epoch=epoch))
