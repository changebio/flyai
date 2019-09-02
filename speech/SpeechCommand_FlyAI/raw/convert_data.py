# -*- coding: utf-8 -*
from collections import Counter

import codecs
import hashlib
import numpy as np
import os
import pandas as pd
import random
import shutil


def convert(from_dir, to_dir, file_out):
    names = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    unk_label = 'unknown'
    silence_label = 'silence'
    name_set = set(names)
    wavs = []
    labels = []
    filename_set = set()
    rep_index = 0
    if not os.path.exists(os.path.join(to_dir, 'audio')):
        os.makedirs(os.path.join(to_dir, 'audio'))
    for root, dirs, filenames in os.walk(from_dir):
        end_index = root.rfind('/')
        root = root[end_index + 1:]
        print(root)
        for filename in filenames:
            if not filename.endswith('.wav'):
                continue
            m2 = hashlib.md5()
            m2.update(codecs.encode(os.path.join(from_dir, 'audio', root, filename)))
            new_filename = m2.hexdigest() + '.wav'
            if new_filename not in filename_set:
                filename_set.add(new_filename)
            else:
                new_filename = str(rep_index) + "_" + new_filename
                rep_index += 1
                filename_set.add(new_filename)
            shutil.copy(os.path.join(from_dir, 'audio', root, filename), os.path.join(to_dir, 'audio', new_filename))
            wavs.append('audio/' + new_filename)
            if root in name_set:
                labels.append(root)
            elif root == '_background_noise_':
                labels.append(silence_label)
            else:
                labels.append(unk_label)

    # balence the data
    counter = Counter()
    counter.update(labels)
    print(counter)
    valid_avg_num = 0
    for name in names:
        valid_avg_num += counter[name]
    valid_avg_num = valid_avg_num // len(names)
    print('avg num', valid_avg_num)
    repeat_num_silence = valid_avg_num // counter[silence_label]
    sample_ratio = valid_avg_num * 1.0 / counter[unk_label]
    new_wavs = []
    new_labels = []
    for wav, label in zip(wavs, labels):
        if label in name_set:
            new_labels.append(label)
            new_wavs.append(wav)
        elif label == silence_label:
            new_labels.extend([label] * repeat_num_silence)
            new_wavs.extend([wav] * repeat_num_silence)
        else:
            if random.random() < sample_ratio:
                new_wavs.append(wav)
                new_labels.append(label)

    data = np.array([new_wavs, new_labels])
    data = data.T
    df = pd.DataFrame(data, columns=['wav', 'label'])
    df.to_csv(file_out, index=False, header=True)
