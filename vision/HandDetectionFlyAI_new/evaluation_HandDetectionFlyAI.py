import sys

import json
import numpy as np
import random
from flyai.dataset import Dataset

from model import Model

dataset = Dataset()
x_test, b_box = dataset.evaluate_data_no_processor(sys.argv[1])

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(x_test)

random.seed(randnum)
random.shuffle(b_box)

model = Model(dataset)

pred = model.predict_all(x_test)
pred_b_box = pred[0]
img_size = pred[1]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


if len(b_box) != len(pred_b_box):
    result = dict()
    result['score'] = 0
    result['label'] = "评估违规"
    result['info'] = ""
    print(json.dumps(result))
else:
    min_iou = np.inf
    for i in range(len(b_box)):
        temp_pred_bb = pred_b_box[i][0]
        true_bb = b_box[i]
        true_bb_y = np.array(
            [true_bb['p3'], true_bb['p5'], true_bb['p7'], true_bb['p9'], true_bb['p11'], true_bb['p13'], true_bb['p15'],
             true_bb['p17'], true_bb['p19'], true_bb['p21'], true_bb['p23'], true_bb['p25']])
        true_bb_x = np.array(
            [true_bb['p4'], true_bb['p6'], true_bb['p8'], true_bb['p10'], true_bb['p12'], true_bb['p14'],
             true_bb['p16'], true_bb['p18'], true_bb['p20'], true_bb['p22'], true_bb['p24'], true_bb['p26']])
        height_scale = true_bb['p1'] / img_size[0]
        width_scale = true_bb['p2'] / img_size[1]

        temp_pred_bb[:12] = (temp_pred_bb[:12] * width_scale).astype(int)
        temp_pred_bb[12:] = (temp_pred_bb[12:] * height_scale).astype(int)
        temp_pred_x = temp_pred_bb[:12]
        temp_pred_y = temp_pred_bb[12:]

        for j in range(6):
            pred_bb = [temp_pred_x[j * 2], temp_pred_y[j * 2], temp_pred_x[j * 2 + 1], temp_pred_y[j * 2 + 1]]
            true_bb = [true_bb_x[j * 2], true_bb_y[j * 2], true_bb_x[j * 2 + 1], true_bb_y[j * 2 + 1]]
            iou = bb_intersection_over_union(pred_bb, true_bb)
            min_iou = min(min_iou, iou)

    result = dict()
    result['score'] = round(min_iou * 100, 2)
    result['label'] = "分数为准确率"
    result['info'] = ""
    print(json.dumps(result))
