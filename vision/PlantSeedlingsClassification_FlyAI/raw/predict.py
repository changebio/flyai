# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
img = 'img/163c13912.png'
p = model.predict(path=img)
print(p)
