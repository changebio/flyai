# -*- coding: utf-8 -*
import argparse
import codecs
import os.path
import tensorflow as tf
from flyai.dataset import Dataset

from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = None

from keras.backend.tensorflow_backend import set_session

feat_type = 'mfcc'  ##'dmfcc'#None  'normmfcc'
label_smooth = False

import sys

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# 超参
parser = argparse.ArgumentParser()
# 111902 约等于512*200
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
args = parser.parse_args()

dataset = Dataset()
model_helper = Model(dataset, do_train=True)


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession(config=config)
    set_session(sess)
    cnn_model = model_helper.model
    cnn_model.train_model(sess, dataset, model_helper, args.BATCH, args.EPOCHS)


main()
