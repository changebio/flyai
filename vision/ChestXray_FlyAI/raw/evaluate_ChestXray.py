import sys

import json
import random
from flyai.dataset import Dataset

from model import Model
print("data are evaluated",sys.argv[1])
dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor(sys.argv[1])

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(x_test)

random.seed(randnum)
random.shuffle(y_test)

model = Model(dataset)
labels = model.predict_all(x_test)
if len(y_test) != len(labels):
    result = dict()
    result['score'] = 0
    result['label'] = "评估违规"
    result['info'] = ""
    print(json.dumps(result))
else:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for index in range(len(labels)):
        label = labels[index]
        test = y_test[index]
        if 'labels' in str(label):
            label = label['labels']

        if label == 0:
            if label == test['labels']:
                tp += 1
            else:
                fp += 1
        elif label == 1:
            if label == test['labels']:
                tn += 1
            else:
                fn += 1
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall / (precision + recall))
    result = dict()
    result['score'] = round(f1 * 100, 2)
    result['label'] = "分数为准确率"
    result['info'] = ""
    print(json.dumps(result))
