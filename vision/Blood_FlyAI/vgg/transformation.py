# -*- coding: utf-8 -*-

class Transformation:
    '''
    处理训练数据的类，某些情况下需要对训练的数据再一次的处理。
    如无需处理的话，不用实现该方法。
    '''

    def transformation_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        if x_train is not None:
            x_train[:, 0, :, :] = (x_train[:, 0, :, :] - 0.485) / 0.229
            x_train[:, 1, :, :] = (x_train[:, 1, :, :] - 0.456) / 0.224
            x_train[:, 2, :, :] = (x_train[:, 2, :, :] - 0.406) / 0.225

        if x_test is not None:
            x_test[:, 0, :, :] = (x_test[:, 0, :, :] - 0.485) / 0.229
            x_test[:, 1, :, :] = (x_test[:, 1, :, :] - 0.456) / 0.224
            x_test[:, 2, :, :] = (x_test[:, 2, :, :] - 0.406) / 0.225

        return x_train, y_train, x_test, y_test