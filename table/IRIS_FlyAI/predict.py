from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(sepal_length=4.3, sepal_width=3.4, petal_length=5.4, petal_width=3.3)
print(p)
