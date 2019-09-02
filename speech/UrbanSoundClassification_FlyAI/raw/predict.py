# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
wav = 'wav/63.wav'
p = model.predict(ID=wav)
print(p)
