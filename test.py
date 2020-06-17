# coding=utf-8
from transformers import BertTokenizer, XLNetModel, AlbertTokenizer, AlbertForQuestionAnswering, ElectraModel, \
    XLNetTokenizer, BertModel, AlbertConfig, BertConfig


def xavier(model, escapes=None, escapekey=None):
    param_num = 0
    escape = 0
    for name, param in model.named_parameters():
        s_ = 1
        for ax_ in param.size():
            s_ *= ax_
        param_num += s_
        if (escapes is not None and name in escapes) or (escapekey is not None and escapekey in name):
            # print('[escape]', name, '[size]', param.size())
            escape += s_
            continue
    print('[total params]', param_num, '[escape]', escape, '[init]', param_num - escape)


def get_offsets(s, pattern):
    base = 0
    offset = []
    s = s.lower().replace('â', 'a')
    for idx, p in enumerate(pattern):
        start_idx, end_idx = find_index(s[base:], pattern, idx)
        offset.append((base + start_idx, base + end_idx))
        base += end_idx
    return offset


def find_index(s, pattern, idx, roll=False):
    pattern[idx] = pattern[idx].lower().replace('▁', '')
    if pattern[idx] == '<unk>' and idx + 1 < len(pattern):
        start, end = find_index(s, pattern, idx + 1, roll=True)
        return 0, start
    elif pattern[idx] == '<unk>':
        return 0, len(s)
    try:
        start = s.index(pattern[idx])
        end = start + len(pattern[idx])
    except ValueError:
        start = 0
        end = 0
    return start, end


def rebuild(raw, offset):
    ans = ''
    for ix in range(len(offset)):
        ans += raw[offset[ix][0]:offset[ix][1]]
        if ix + 1 < len(offset) and offset[ix][1] < offset[ix + 1][0]:
            ans += ' '
    return ans


data = []
with open('data/train.csv', 'r') as f:
    for line in f:
        data.append(line.split(',')[2])

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
tokenizer.save_vocabulary('.')
input('done')
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
print(len(tokenizer))
tokenizer.save_vocabulary('xlnet_model')
input('DONE')
print(len(tokenizer))
model = XLNetModel.from_pretrained('xlnet-base-cased', output_hidden_states=True, output_attentions=True)
model.save_pretrained('xlnet_model')
input('DONE')
# model = AlbertForQuestionAnswering.from_pretrained('albert-large-v2')
# xavier(model)
# input()
raw_text = ":''>ï¿½-^â?~"
# raw_text = line
raw_text = raw_text.replace('ï¿½', '').replace('ï', 'i').replace('``', '"').replace('^', '#').replace("''", '"')
raw = ' '.join(raw_text.split())
raw_text = raw.replace('-', '#')
encode = tokenizer.encode_plus('positive', raw_text.lower())['input_ids']
encode = [x for x in encode]
print(encode)
tokens = tokenizer.convert_ids_to_tokens(encode)
print(tokens)
aa = get_offsets(raw_text, tokens)
print(aa)
print(rebuild(raw, aa))
print(raw)
input()
for line in data:
    # raw_text = 'my guys call me ``BartÂ ofÂ theÂ criticalÂ questions``. I`m guessing that`s a good thing.  #zeropoint.IT'
    raw_text = line
    raw_text = raw_text.replace('ï¿½', '').replace('ï', 'i').replace('``', '"').replace('^', '#').replace("''", '"')
    raw = ' '.join(raw_text.split())
    raw_text = raw.replace('-', '#')
    encode = tokenizer.encode(raw_text)
    encode = [x for x in encode if x != 13]
    tokens = tokenizer.convert_ids_to_tokens(encode)
    # print(tokens)
    aa = get_offsets(raw_text, tokens)
    # print(aa)
    if rebuild(raw, aa).strip() != raw.strip():
        print(raw)
    # print(rebuild(raw_text, aa))
# #
# from transformers import BertTokenizer, BertForQuestionAnswering, ElectraModel
# import torch
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = ElectraModel.from_pretrained('google/electra-base-discriminator')
# xavier(model)
# question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
# encoding = tokenizer.encode_plus(question, text)
# input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
# start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
#
# all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
# answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
#
# assert answer == "a nice puppet"
