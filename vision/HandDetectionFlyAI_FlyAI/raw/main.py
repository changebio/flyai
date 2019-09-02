# -*- coding: utf-8 -*
import argparse
import tensorflow as tf
from flyai.dataset import Dataset
import numpy as np
from path import MODEL_PATH
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
parser.add_argument("-t", "--TRAINLAYER", default='fc8,fc7', type=str, help="train layer")
parser.add_argument("-lr", "--LEARNINGRATE", default=0.001, type=float, help="batch size")
parser.add_argument("-kp", "--KEEPPROB", default=0.5, type=float, help="dropout keep prob")
args = parser.parse_args()

# 构建模型
img_size = [224, 224]
class_num = 24  # 6个框 * 两个坐标对应的4个值

x = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
y = tf.placeholder(tf.float32, [None, class_num])
dropout_keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)

data = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.BATCH)
model = Model(data)
model.create_model(class_num, dropout_keep_prob)

loss = model.vgg.loss(x, y)
train_layers = args.TRAINLAYER.split(',')
train_op = model.vgg.optimize(args.LEARNINGRATE, global_step, train_layers)
tf.summary.scalar('train_loss', loss)
merged_summary = tf.summary.merge_all()


saver = tf.train.Saver()
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session() as sess:
    sess.run(init)
    pretrain_model_path = model.get_remote_date("https://dataset.flyai.com/vgg16_weights.npz")
    model.vgg.load_original_weights(sess, skip_layers=train_layers, pretrain_model=pretrain_model_path)
    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        print('find model !')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('can not find checkpoint!')

    min_val_loss = np.inf
    for i in range(data.get_step()):
        x_train, y_train = data.next_train_batch()
        x_val, y_val = data.next_validation_batch()
        _, train_loss, step = sess.run([train_op, loss, global_step], feed_dict={x: x_train, y: y_train, dropout_keep_prob: args.KEEPPROB})
        print('train step: %d, loss: %f' % (step, train_loss))
        #  val
        if(step % 10 == 0):
            val_loss = sess.run(loss, feed_dict={x: x_val, y: y_val, dropout_keep_prob: 1.})
            print('---------------val loss: %f, min val loss: %f' % (val_loss, min_val_loss))
            if(val_loss<min_val_loss):
                min_val_loss = val_loss
                model.save_model([sess, saver,step], MODEL_PATH)
                print('---------------model save step %d suceess' % step)





