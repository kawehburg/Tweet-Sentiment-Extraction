import numpy as np
import random


def get_file(filename):
    _data = []
    with open(filename, 'r', encoding='utf-8') as fs:
        for _line in fs:
            _data.append(_line[:-1])
    return _data


raw = get_file('data/train.csv')[1:]
k_num = [i for i in range(len(raw))]
fold_num = {}
fold = 0
random.shuffle(k_num)
for i in range(len(k_num)):
    fold_num[k_num[i]] = fold
    fold = (fold + 1) % 8

with open('data/train8_folds.csv', 'w', encoding='utf-8') as f:
    f.write('textID,text,selected_text,sentiment,kfold\n')
    for i, line in enumerate(raw):
        if line.split(',')[-2] != '':
            f.write(line + ',' + str(fold_num[i]) + '\n')
