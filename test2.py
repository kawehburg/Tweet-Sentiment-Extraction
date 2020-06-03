import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_sub(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            ref = ','.join(line[:-1].split(',')[1:])
            data.append(ref)
    return data[1:]


sub1 = get_sub('results/folds_submission.csv')
# train_xlnet_768_span_cnn_submission = get_sub('results/train_xlnet_768_span_cnn_submission.csv')
sub2 = get_sub('results/folds_old_submission.csv')
jcd = []
for ref1, ref2 in zip(sub1, sub2):
    jcd.append(jaccard(ref1, ref2))
    if jcd[-1] < 0.5:
        print('##########')
        print(ref1)
        print(ref2)

print(sum(jcd) / len(jcd))
