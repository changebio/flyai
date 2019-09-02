# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
from keras.engine.saving import load_model

from path import MODEL_PATH

KERAS_MODEL_NAME = "model.h5"


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset

    def predict(self, **data):
        model = load_model(os.path.join(MODEL_PATH, KERAS_MODEL_NAME))
        data = model.predict(self.dataset.predict_data(**data))
        data = self.dataset.to_categorys(data)
        return data

    def predict_all(self, datas):
        model = load_model(os.path.join(MODEL_PATH, KERAS_MODEL_NAME))
        stars = []
        for data in datas:
            data = model.predict(self.dataset.predict_data(**data))
            data = self.dataset.to_categorys(data)
            stars.append(data)
        return stars

    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        print(os.path.join(path, name))
        model.save(os.path.join(path, name))
