from transformers import BertTokenizer, XLNetTokenizer, AlbertTokenizer, AlbertForQuestionAnswering, ElectraModel


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
with open('data/test.csv', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(line.split(',')[2])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(len(tokenizer))
model = ElectraModel.from_pretrained('google/electra-large-discriminator', output_hidden_states=True, output_attentions=True)
# model = AlbertForQuestionAnswering.from_pretrained('albert-large-v2')
# xavier(model)
# input()
# raw_text = 'BABEE. ) LOVE YOOOOUUU. > RP time."'
# # raw_text = line
# raw_text = raw_text.replace('ï¿½', '').replace('ï', 'i').replace('``', '"').replace('\'', '"').replace('`', '\'')
# raw_text = ' '.join(raw_text.split())
# encode = tokenizer.encode(raw_text.lower())
# encode = [x for x in encode if x != 13]
# print(encode)
# tokens = tokenizer.convert_ids_to_tokens(encode)[1:-1]
# print(tokens)
# aa = get_offsets(raw_text, tokens)
# print(aa)
# print(rebuild(raw_text, aa))
# print(raw_text)
# input()
for line in data:
    # raw_text = 'my guys call me ``BartÂ ofÂ theÂ criticalÂ questions``. I`m guessing that`s a good thing.  #zeropoint.IT'
    raw_text = line
    raw_text = raw_text.replace('ï¿½', '').replace('ï', 'i').replace('``', '"').replace('\'', '"').replace('`', '\'')
    raw_text = ' '.join(raw_text.split())
    encode = tokenizer.encode(raw_text)
    encode = [x for x in encode if x != 13]
    tokens = tokenizer.convert_ids_to_tokens(encode)[1:-1]
    # print(tokens)
    aa = get_offsets(raw_text, tokens)
    # print(aa)
    if rebuild(raw_text, aa).strip() != raw_text.strip():
        print(raw_text)
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