import sys

import json
import random
from flyai.dataset import Dataset
from nltk.translate.bleu_score import sentence_bleu

from model import Model

dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor(sys.argv[1])

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(x_test)

random.seed(randnum)
random.shuffle(y_test)

x_test = x_test[0:10000]
y_test = y_test[0:10000]

model = Model(dataset)
labels = model.predict_all(x_test)
eval = 0

if len(y_test) != len(labels):
    result = dict()
    result['score'] = 0
    result['label'] = "评估违规"
    result['info'] = ""
    print(json.dumps(result))
else:
    sum_sorce = 0
    for index in range(len(labels)):
        l = list(labels[index]['answer'])
        a = list(y_test[index]['answer'])
        index = len(a)
        if len(a) > len(l):
            index = min(len(a), len(l) + 5)  # 由于模型输出有限制，当标准答案过长时，剪切掉一部分
        a_cut = a[:index]
        score = sentence_bleu([l], a_cut)
        if l == a:
            score = 1.0
        sum_sorce += score
    sum_sorce = sum_sorce / len(labels)
    result = dict()
    result['score'] = round(sum_sorce * 100, 2)
    result['label'] = "分数为准确率"
    result['info'] = ""
    print(json.dumps(result))
