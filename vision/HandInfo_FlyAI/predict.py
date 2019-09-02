# -*- coding: utf-8 -*
'''
实现模型的预测
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(image_path="/Users/lijiayi/Documents/cece_python/Facevividetection/data/input/6.png")
print(p)
