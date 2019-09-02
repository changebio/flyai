# -*- coding: utf-8 -*
import os
import tensorflow as tf
from flyai.model.base import Base

import vgg
from path import MODEL_PATH

TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data

    def create_model(self, class_num, dropout_keep_prob):
        self.vgg = vgg.VggNetModel(num_classes=class_num, dropout_keep_prob=dropout_keep_prob)

    def save_model(self, sess_info, path, name=TENSORFLOW_MODEL_DIR, overwrite=False):
        self.check(path, overwrite)
        sess = sess_info[0]
        saver = sess_info[1]
        step = sess_info[2]
        saver.save(sess, os.path.join(path, name), global_step=step)

    # 在 predict 和 predict_all 中，需要返回img_size
    def predict(self, **data):
        img_size = [224, 224]
        class_num = 24
        x = tf.placeholder(tf.float32, [1, img_size[0], img_size[1], 3])
        self.create_model(class_num, 1.)
        get_prob_bb = self.vgg.inference(x)
        # 坐标映射
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        with tf.Session() as session:
            # 加载参数
            saver.restore(session, ckpt.model_checkpoint_path)
            x_data = self.data.predict_data(**data)
            logits = session.run(get_prob_bb, feed_dict={x: x_data})
            logits[:, :12] = logits[:, :12] * img_size[1]
            logits[:, 12:] = logits[:, 12:] * img_size[0]
            return [logits, img_size]

    def predict_all(self, datas):
        img_size = [224, 224]
        class_num = 24
        x = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        self.create_model(class_num, 1.)
        get_prob_bb = self.vgg.inference(x)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        with tf.Session() as session:
            saver.restore(session, ckpt.model_checkpoint_path)
            outputs = []
            for data in datas:
                x_data = self.data.predict_data(**data)
                predict = session.run(get_prob_bb, feed_dict={x: x_data})
                predict[:, :12] = predict[:, :12] * img_size[1]  #
                predict[:, 12:] = predict[:, 12:] * img_size[0]
                outputs.append(predict)
            return [outputs, img_size]
