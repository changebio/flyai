# -*- coding: utf-8 -*
import os
import tensorflow as tf
from flyai.model.base import Base
from path import MODEL_PATH
from vgg import VggNetModel
from config import class_num, image_resize, output_size, width_len, height_len
import numpy as np
from flyai.processor.download import check_download
from path import DATA_PATH, OUTPUT_PATH
import cv2

TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.model_path = MODEL_PATH
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.session = tf.Session()
        self.is_load = False
        self.create_model()
        self.saver = tf.train.Saver()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        tf.global_variables_initializer().run(session=self.session)
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('load ckpt model !!!!!')
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print('can not find checkpoint')

    def create_model(self):
        self.x = tf.placeholder(tf.float32, [None, image_resize[0], image_resize[1], 3], name='input_x')
        self.y = tf.placeholder(tf.float32, [None, output_size[0], output_size[1], class_num + 4], name='input_y')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.vgg = VggNetModel(class_num)
        self.get_loss = self.vgg.loss(self.x, self.y)
        self.train_op = self.vgg.optimize(self.lr)

    # 把经过模型预测的结果，按照resize 的比例 返回到相对于原图大小中
    def decode_pred_bb(self, one_img_y, scale):
        one_result = []
        one_score = []
        pred_class_label = np.argmax(one_img_y[0, :, :, :-4], axis=-1)  # (32, 32)
        #  后续可以在这里加上极大值抑制（这里没写）
        for i in range(output_size[0]):  # h
            for j in range(output_size[1]):  # w
                if (pred_class_label[i][j] != 0 and one_img_y[0, i,  j, pred_class_label[i][j]] > 0.5):
                    temp_pred = one_img_y[0, i, j, -4:]  #
                    temp_width = int(round(np.exp(temp_pred[2]) * scale[0]))
                    temp_height = int(round(np.exp(temp_pred[3]) * scale[1]))

                    temp_left = int(
                        round((temp_pred[0] + j) * width_len * scale[0])) - temp_width // 2
                    temp_top = int(
                        round((temp_pred[1] + i) * height_len * scale[1])) - temp_height // 2
                    # <class_name> <left> <top> <right> <bottom>
                    one_result.append([pred_class_label[i][j], temp_left, temp_top, temp_left + temp_width,
                                       temp_top + temp_height])
                    one_score.append(one_img_y[0, i,  j, pred_class_label[i][j]])

        one_result = np.array(one_result)
        one_score = np.array(one_score)
        # 这里按照socre进行排序，并把排序好的预测用于计算map
        one_result = one_result[(one_score*-1).argsort()]
        return one_result

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        # 这里没有
        data = self.data.predict_data(**data)
        images = data[0]
        scale = data[1:]
        conv_final = self.session.run(self.vgg.conv_final, feed_dict={self.x : images}) # (512, 512, 11+4)
        # 对返回的结果进行处理
        preds_bb = self.decode_pred_bb(conv_final, scale)
        return preds_bb

    def predict_all(self, datas):
        labels = []
        for data in datas:
            print(data)
            path = check_download(data['image_path'], DATA_PATH)
            images = cv2.imread(path)
            # 获得 缩放 比例
            height, width, _ = images.shape
            height_scale = height / image_resize[0]
            width_scale = width / image_resize[1]

            data = self.data.predict_data(**data) # (1,512, 512, 3)
            conv_final = self.session.run(self.vgg.conv_final, feed_dict={self.x: data}) # (1, 32, 32, 19)
            print(conv_final)
            preds_bb = self.decode_pred_bb(conv_final, [width_scale, height_scale])
            print(preds_bb)
            labels.append(preds_bb)
        return labels



    def get_tensor_name(self, name):
        return name + ":0"

    def delete_file(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    # 关闭连接
    def close_session(self):
        self.session.close()


