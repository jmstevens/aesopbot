import sys
import os
from src.features.build import Dataset, Provider
from src.models.rnn_model import RNNModel #.rnn_model import RNNModel
from tensorflow.contrib.tensorboard.plugins import projector
import sys
import numpy as np
import time
import tensorflow as tf
import copy
from tqdm import tqdm
import glob
import csv

data_dir = "src/data"
tensorboard_dir = data_dir + "/tensorboard/" + str(time.strftime("%Y-%m-%d"))
input_file = data_dir + "/input.txt"
output_file = data_dir + "/output.txt"
output = open(output_file, "w")
output.close()


class Freestyle:
    BATCH_SIZE = 25
    SEQUENCE_LENGTH = 12
    STARTER_LEARNING_RATE = 0.001
    DECAY_RATE = 0.95
    HIDDEN_LAYER_SIZE = 512
    CELLS_SIZE = 2
    TRAIN_KEEP_PROB = 0.2
    GRADIENT_CLIP = 5.
    USE_PEEPHOLES = False
    TOTAL_EPOCHS = 1000
    # DROPOUT = 0.9

    TEXT_SAMPLE_LENGTH = 500
    SAMPLING_FREQUENCY = 1
    LOGGING_FREQUENCY = 1000

    data_dir = "src/data"
    tensorboard_dir = data_dir + "/tensorboard/" + str(time.strftime("%Y-%m-%d"))
    input_file = data_dir + "/input.txt"
    output_file = data_dir + "/output.txt"
    save_dir = os.path.join(data_dir,"aesop.ckpt")

    def __init__(self, model_load_path, prime_text, training=True):
        self.sess = tf.Session()
        self.training=training

        print("Process data.....")
        self.provider = Provider(self.BATCH_SIZE, self.SEQUENCE_LENGTH)
        self.vocabulary = self.provider.vocabulary
        # dict_writer = csv.DictWriter(os.path.join(self.save_dir, 'metadata.tsv'), keys, delimiter='\t')

        print("Initializing model....")
        self.model = RNNModel(self.sess,
                              self.vocabulary,
                              self.BATCH_SIZE,
                              self.SEQUENCE_LENGTH,
                              self.HIDDEN_LAYER_SIZE,
                              self.CELLS_SIZE,
                              self.TRAIN_KEEP_PROB,
                              self.GRADIENT_CLIP,
                              self.STARTER_LEARNING_RATE,
                              self.USE_PEEPHOLES,
                              self.training
                              )

        print("Initialize variables.....")
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

        if model_load_path:
            _num = str(max([int(i.replace('ckpt-','')) for i in list(set([i.split('.')[1] for i in os.listdir("src/data") if 'aesop.ckpt-' in i]))]))
            self.saver.restore(self.sess, "src/data/aesop.ckpt-{}".format(_num))
            print("Model restored from {}".format(self.save_dir))

        if self.training:
            self.train()
        else:
            print("foo")
            self.test(prime_text)

    def train(self):
        print("Training model..........")
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.tensorboard_dir)
        writer.add_graph(self.sess.graph)

        # epoch = 0
        temp_losses = []
        smooth_losses = []
        temp_accuracy = []
        smooth_accuracies = []
        # while True:
        for epoch in range(1, self.TOTAL_EPOCHS + 1):
            bar = tqdm(range(1, self.provider.batches_size + 1))
            self.provider = Provider(self.BATCH_SIZE, self.SEQUENCE_LENGTH)
            # self.provider.reset_batch_pointer()
            for batch in bar:
                inputs, targets = self.provider.next_batch()
                feed_dict = {self.model.input_data: inputs, self.model.targets: targets}
                summary, global_step, loss, accuracy, _ = self.sess.run([summaries,
                                                                            self.model.global_step,
                                                                            self.model.loss,
                                                                            self.model.accuracy,
                                                                            self.model.train_op],
                                                                         feed_dict=feed_dict)
                bar.set_description("Epoch:{0} | Global Step:{1} | Batch Number: {2} | Loss: {3} | Accuracy: {4}".format(epoch, global_step, batch, loss, accuracy))
                bar.refresh()
                writer.add_summary(summary, global_step)

            temp_losses.append(loss)
            smooth_loss = np.mean(temp_losses)
            smooth_losses.append(smooth_loss)
            print('{{"metric": "average loss", "value": {}}}'.format(smooth_loss))
            temp_losses = []

            temp_accuracy.append(accuracy)
            smooth_accuracy = np.mean(temp_accuracy)
            smooth_accuracies.append(smooth_accuracy)
            print('{{"metric": "average accuracy", "value": {}}}'.format(accuracy))
            temp_losses = []

            # embedding_conf.tensor_name = embeddings
            # writer.add_summary(summary, global_step)
            print("Saving model......")
            self.saver.save(self.sess, self.save_dir, global_step=global_step)
            meta = [i for i in os.listdir(self.data_dir) if '.meta' in i][0]
            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            # embedding.tensor_name = summary.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = os.path.join(self.save_dir, meta)
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(self.save_dir), config)

    def test(self, prime_text):
        sample = self.model.generate(priming_text=prime_text)
        return sample

def main():
    if not glob.glob('src/data/*.ckpt*'):
        load_path = None
    else:
        load_path = 'yes'
    training = True
    prime_text = None

    Freestyle(load_path, prime_text, training)

if __name__ == '__main__':
    main()
