# -*- coding: utf-8 -*
'''
实现模型的预测
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(image_path='images/00008/Deng.ttf.png')
print(p)
