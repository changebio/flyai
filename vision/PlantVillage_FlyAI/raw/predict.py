import os
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(image_path=os.path.join('images', 'cifar10_test_259_9.jpg'))
print(p)
