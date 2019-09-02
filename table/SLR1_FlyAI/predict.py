# -*- coding: utf-8 -*
'''
实现模型的预测
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(path="pos/P01_11_09_1_body.txt")
print(p)
