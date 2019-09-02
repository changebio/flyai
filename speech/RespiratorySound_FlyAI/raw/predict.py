# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data, do_train=False)
p = model.predict(audio_and_txt_files_path='audio_and_txt_files/161.zip',
                  patient_id=160,
                  age=3.0,
                  sex='F',
                  adult_bmi=None,
                  child_weight=19.0,
                  child_height=99.0)
print(data.to_categorys(p))
