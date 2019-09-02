# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data, do_train=False)
wav = 'audio/8ce19853dcf57ebd430abffb6efe529d.wav'
p = model.predict(wav=wav)
print(data.to_categorys(p))
