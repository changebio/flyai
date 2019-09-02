# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)

img_path = 'CNY/0HI6RGPO.jpg'
p = model.predict(image_path=img_path)
print(p)
