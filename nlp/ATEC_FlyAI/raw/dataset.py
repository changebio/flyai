# -*- coding: utf-8 -*

#import gevent
import numpy
import os
#from gevent import pool, monkey

from flyai.source.source import Source
from flyai.utils.yaml_helper import Yaml

#monkey.patch_all(ssl=False)

#pool = pool.Pool(128)


class Dataset:
    def __init__(self, epochs=5, batch=32, val_batch=32, transformation=None, source=Source(),
                 processor_pka="processor",
                 processor_class=None, config_path=None):
        self.__db = source
        if epochs > 15:
            epochs = 15
        if batch > 256:
            batch = 256
        self.__BATCH = batch
        self.__VAL_BATCH = val_batch
        self.__STEP = round(self.__db.get_train_length() / batch) * epochs
        if config_path is not None:
            self.__model = Yaml(path=os.path.join(config_path, "app.yaml")).processor()
        else:
            self.__model = Yaml().processor()
        self.test_trans = False
        if processor_class is not None:
            clz = processor_class
        else:
            clz = self.__model['processor']

        if config_path is not None:
            config = config_path.replace(os.path.sep, '.')
            processor_pka = config + "." + processor_pka
            self.processor = self.create_instance(processor_pka, clz)
        else:
            self.processor = self.create_instance(processor_pka, clz)

        if transformation is not None:
            self.transformation = transformation
        else:
            try:
                if config_path is not None:
                    config = config_path.replace(os.path.sep, '.')
                    self.transformation = self.create_instance(config + '.transformation', 'Transformation')
                else:
                    self.transformation = self.create_instance('transformation', 'Transformation')
            except:
                self.transformation = None

    def get_step(self):
        return self.__STEP

    def get_train_length(self):
        return self.__db.get_train_length()

    def get_validation_length(self):
        return self.__db.get_validation_length()

    def next_train_batch(self):
        x_train, y_train = self.__db.next_train_batch(self.__BATCH)
        x_train = self.processor_x(x_train)
        y_train = self.processor_y(y_train)
        return x_train, y_train

    def next_validation_batch(self):
        x_val, y_val = self.__db.next_validation_batch(self.__VAL_BATCH)
        x_val = self.processor_x(x_val)
        y_val = self.processor_y(y_val)
        return x_val, y_val

    def next_batch(self, size=32, test_size=32, test_data=True):
        x_train, y_train, x_val, y_val = self.__db.next_batch(size, test_size)
        x_train = self.processor_x(x_train)
        y_train = self.processor_y(y_train)
        if test_data:
            x_val = self.processor_x(x_val)
            y_val = self.processor_y(y_val)
        else:
            x_val = None
            y_val = None
        if self.transformation is not None:
            if self.test_trans:
                x_train, y_train, _, _ = self.transformation.transformation_data(x_train, y_train)
            else:
                x_train, y_train, x_val, y_val = self.transformation.transformation_data(x_train, y_train, x_val,
                                                                                         y_val)
                self.test_trans = True
        return x_train, y_train, x_val, y_val

    def evaluate_source(self, path):
        x_val, y_val = self.__db.get_evaluate_data(path)
        return x_val, y_val

    def evaluate_data(self, path=None):
        if path is None:
            x_val, y_val = self.__db.get_validation_data()
        else:
            x_val, y_val = self.__db.get_evaluate_data(path)
        x_val = self.processor_x(x_val)
        y_val = self.processor_y(y_val)
        if self.transformation is not None:
            _, _, x_val, y_val = self.transformation.transformation_data(x_test=x_val, y_test=y_val)
        return x_val, y_val

    def evaluate_data_no_processor(self, path=None):
        if path is None:
            x_val, y_val = self.__db.get_validation_data()
        else:
            x_val, y_val = self.__db.get_evaluate_data(path)
        return x_val, y_val

    def get_all_data(self):
        x_train, y_train, x_val, y_val = self.__db.get_all_data()
        return x_train, y_train, x_val, y_val

    def get_all_validation_data(self):
        _, _, x_val, y_val = self.get_all_data()
        x_val = self.processor_x(x_val)
        y_val = self.processor_y(y_val)
        return x_val, y_val

    def get_all_processor_data(self):
        x_train, y_train, x_val, y_val = self.get_all_data()
        x_train = self.processor_x(x_train)
        y_train = self.processor_y(y_train)
        x_val = self.processor_x(x_val)
        y_val = self.processor_y(y_val)
        if self.transformation is not None:
            x_train, y_train, x_val, y_val = self.transformation.transformation_data(x_train, y_train, x_val, y_val)
        return x_train, y_train, x_val, y_val

    def processor_x(self, x_datas):
        processors = []
        init = False
        processor_len = 0
        for data in x_datas:
            processor = self.get_method_dict(self.processor, self.__model['input_x'], **data)
            if not isinstance(processor, tuple):
                processor_len = 1
            if processor_len == 1:
                processors.append(numpy.array(processor))
            else:
                if not init:
                    processors = [[] for i in range(len(processor))]
                    init = True
                index = 0
                for item in processor:
                    processors[index].append(numpy.array(item))
                    index += 1
        if processor_len == 1:
            return numpy.concatenate([processors], axis=0)
        else:
            list = []
            for column in processors:
                list.append(numpy.concatenate([column], axis=0))
            return list

    def processor_y(self, y_datas):
        processors = []
        init = False
        processor_len = 0
        for data in y_datas:
            processor = self.get_method_dict(self.processor, self.__model['input_y'], **data)
            if not isinstance(processor, tuple):
                processor_len = 1
            if processor_len == 1:
                processors.append(numpy.array(processor))
            else:
                if not init:
                    processors = [[] for i in range(len(processor))]
                    init = True
                index = 0
                for item in processor:
                    processors[index].append(numpy.array(item))
                    index += 1
        if processor_len == 1:
            return numpy.concatenate([processors], axis=0)
        else:
            list = []
            for column in processors:
                list.append(numpy.concatenate([column], axis=0))
            return list

    def predict_data(self, **data):
        processors = []
        processor = self.get_method_dict(self.processor, self.__model['output_x'], **data)
        processor_len = 0
        if not isinstance(processor, tuple):
            processors.append(numpy.array(processor))
            processor_len = 1
        else:
            processors = [[] for i in range(len(processor))]
            index = 0
            for item in processor:
                processors[index].append(numpy.array(item))
                index += 1
        if processor_len == 1:
            x = numpy.concatenate([processors], axis=0)
        else:
            list = []
            for column in processors:
                list.append(numpy.concatenate([column], axis=0))
            x = list
        if self.transformation is not None:
            x, _, _, _ = self.transformation.transformation_data(x)
        return x

    def to_categorys(self, predict):
        return self.get_method_list(self.processor, self.__model['output_y'], predict)

    def create_instance(self, module_name, class_name, *args, **kwargs):
        module_meta = __import__(module_name, globals(), locals(), [class_name])
        class_meta = getattr(module_meta, class_name)
        return class_meta(*args, **kwargs)

    def get_method_dict(self, clz, method_name, **args):
        m = getattr(clz, method_name)
        return m(**args)

    def get_method_list(self, clz, method_name, *args):
        m = getattr(clz, method_name)
        return m(*args)
