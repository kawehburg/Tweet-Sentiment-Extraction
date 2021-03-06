import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from torch.nn.init import xavier_uniform_
from torch.nn.modules.normalization import LayerNorm
from torch.autograd import Variable
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import ElectraModel, BertModel, AlbertModel, XLNetModel
from transformers import BertTokenizer, AlbertTokenizer, XLNetTokenizer
from tqdm.autonotebook import tqdm
from .lovas import lovasz_hinge_flat
import copy
import math
import utils


def seed_everything(seed_value):
    # random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        p = self.pe[:x.size(-2)]
        for i in range(len(x.size()) - 2):
            p = p.unsqueeze(0)
        x = x + p
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(30522, 256, padding_idx=0)
        self.pos_embedding = PositionalEmbedding(256)
    
    def forward(self, ids, mask, token_type_ids):
        out = self.pos_embedding(self.embedding(ids))
        return out,


class RoBERTaModel(nn.Module):
    def __init__(self, name='roberta', **kwargs):
        super().__init__()
        model_config = transformers.RobertaConfig.from_pretrained(name)
        # Output hidden states
        # This is important to set since we want to concatenate the hidden states from the last 2 BERT layers
        model_config.output_hidden_states = True
        self.roberta = transformers.RobertaModel.from_pretrained(name, config=model_config)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        return out


class ELECTRAModel(nn.Module):
    def __init__(self, name='google/electra-large-discriminator', **kwargs):
        super().__init__()
        self.bert = ElectraModel.from_pretrained(name, output_hidden_states=True, output_attentions=True)
    
    def forward(self, ids, mask, token_type_ids):
        mask = ids.ne(0).detach()
        out = self.bert(
            ids,
            attention_mask=mask,
        )  # bert_layers x bs x SL x (768 * 2)
        out = out[1]
        return out


class BERTModel(nn.Module):
    def __init__(self, name='bert-base-uncased', **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(name, output_hidden_states=True, output_attentions=True)
    
    def forward(self, ids, mask, token_type_ids):
        mask = ids.ne(0).detach()
        out = self.bert(
            ids,
            attention_mask=mask,
        )  # bert_layers x bs x SL x (768 * 2)
        out = out[2]
        return out


class Albert(nn.Module):
    def __init__(self, name='albert-xxlarge-v2', pad=0, **kwargs):
        super().__init__()
        self.bert = AlbertModel.from_pretrained(name, output_hidden_states=True, output_attentions=True)
        self.pad = pad
    
    def forward(self, ids, mask, token_type_ids):
        mask = ids.ne(self.pad).detach()
        out = self.bert(
            ids,
            attention_mask=mask,
        )  # bert_layers x bs x SL x (768 * 2)
        out = out[2]
        return out


class XLNet(nn.Module):
    def __init__(self, name='xlnet-large-cased', pad=0, **kwargs):
        super().__init__()
        self.bert = XLNetModel.from_pretrained(name, output_hidden_states=True, output_attentions=True)
        self.pad = pad
    
    def forward(self, ids, mask, token_type_ids):
        mask = ids.ne(self.pad).float().detach()
        out = self.bert(
            ids,
            attention_mask=mask,
        )  # bert_layers x bs x SL x (768 * 2)
        out = out[1]
        return out


class Model(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.head = head
    
    def forward(self, ids, mask, token_type_ids):
        out = self.base_model(ids, mask, token_type_ids)
        output = self.head(out, mask)
        return output


class RoBERTaLoader:
    @staticmethod
    def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
        tweet = " " + " ".join(str(tweet).split())
        selected_text = " " + " ".join(str(selected_text).split())
        
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None
        
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
        
        tok_tweet = tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids
        tweet_offsets = tok_tweet.offsets
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        try:
            targets_start = target_idx[0]
            targets_end = target_idx[-1]
        except:
            targets_start = 0
            targets_end = len(input_ids_orig)
            print('>>', tweet)
            print('>>', selected_text)
        
        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }
        
        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4
        
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets,
        }
    
    def __init__(self, tweet, sentiment, selected_text, tweet_id=None, name='roberta', max_len=192, **kwargs):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        ROBERTA_PATH = name
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=f"{ROBERTA_PATH}/vocab.json",
            merges_file=f"{ROBERTA_PATH}/merges.txt",
            lowercase=True,
            add_prefix_space=True
        )
        self.max_len = max_len
        self.tweet_id = tweet_id
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        data = self.process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            'tweet_id': self.tweet_id[item],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


class BERTLoader:
    @staticmethod
    def get_offsets(s, pattern, ignore=0):
        base = 0
        offset = []
        s = s.lower().replace('â', 'a')
        # print(s)
        for i, p in enumerate(pattern):
            if i < ignore:
                offset.append((0, 0))
                continue
            p = p.lower().replace('##', '')
            try:
                start_idx = s[base:].index(p)
                end_idx = start_idx + len(p)
            except ValueError:
                start_idx = 0
                end_idx = 0
            offset.append((base + start_idx, base + end_idx))
            base += end_idx
        
        return offset
    
    @staticmethod
    def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
        # handle special example
        tweet = tweet.replace('ï¿½', '').replace('¿½t', '').replace('ï', '').replace(
            'WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO',
            'WOOO')
        selected_text = selected_text.replace('ï¿½', '').replace('¿½t', '').replace('ï', '').replace(
            'WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO',
            'WOOO')
        
        tweet = " " + " ".join(str(tweet).split())
        selected_text = " " + " ".join(str(selected_text).split())
        question, text = sentiment, tweet
        encoding = tokenizer.encode_plus(question, text)
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        mask = [1] * len(token_type_ids)
        tweet_offsets = BERTLoader.get_offsets(tweet, tokenizer.convert_ids_to_tokens(input_ids), ignore=2)
        
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None
        
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        try:
            targets_start = target_idx[0]
            targets_end = target_idx[-1]
        except:
            targets_start = 3
            targets_end = len(input_ids)
            print('>>', tweet)
            print('>>', selected_text)
        
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets,
        }
    
    def __init__(self, tweet, sentiment, selected_text, tweet_id=None, max_len=192, name='bert-base-uncased'):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        try:
            self.tokenizer = BertTokenizer.from_pretrained(name)
        except OSError:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.tweet_id = tweet_id
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        data = self.process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            'tweet_id': self.tweet_id[item],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


class AlbertLoader:
    @staticmethod
    def get_offsets(s, pattern, ignore=0):
        base = 0
        offset = []
        s = s.lower().replace('â', 'a')
        for idx, p in enumerate(pattern):
            if idx < ignore:
                offset.append((0, 0))
                continue
            start_idx, end_idx = AlbertLoader.find_index(s[base:], pattern, idx)
            offset.append((base + start_idx, base + end_idx))
            base += end_idx
        return offset
    
    @staticmethod
    def find_index(s, pattern, idx, roll=False):
        pattern[idx] = pattern[idx].lower().replace('▁', '')
        if pattern[idx] == '<unk>' and idx + 1 < len(pattern):
            start, end = AlbertLoader.find_index(s, pattern, idx + 1, roll=True)
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
    
    @staticmethod
    def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
        # handle special example
        tweet = tweet.replace('ï¿½', 'i').replace('ï', 'i').replace('``', '"').replace('\'', '"').replace('`', '\'')
        selected_text = selected_text.replace('ï¿½', 'i').replace('ï', 'i').replace('``', '"').replace('\'', '"') \
            .replace('`', '"')
        
        tweet = " " + " ".join(str(tweet).split())
        selected_text = " " + " ".join(str(selected_text).split())
        question, text = sentiment, tweet
        encoding = tokenizer.encode_plus(question, text)
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        mask = [1] * len(token_type_ids)
        tweet_offsets = AlbertLoader.get_offsets(tweet, tokenizer.convert_ids_to_tokens(input_ids)[1:-1], ignore=1)
        tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]
        
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None
        
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        try:
            targets_start = target_idx[0]
            targets_end = target_idx[-1]
        except:
            targets_start = 2
            targets_end = len(input_ids)
            print('>>', tweet)
            print('>>', selected_text)
        
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets,
        }
    
    def __init__(self, tweet, sentiment, selected_text, tweet_id=None, name='albert-xxlarge-v2', max_len=192):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = AlbertTokenizer.from_pretrained(name)
        self.max_len = max_len
        self.tweet_id = tweet_id
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        data = self.process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            'tweet_id': self.tweet_id[item],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


class SP_XLNetLoader:
    @staticmethod
    def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
        tweet = " " + " ".join(str(tweet).split())
        selected_text = " " + " ".join(str(selected_text).split())
        
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None
        
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
        
        tok_tweet = tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids
        tweet_offsets = tok_tweet.offsets
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        try:
            targets_start = target_idx[0]
            targets_end = target_idx[-1]
        except:
            targets_start = 0
            targets_end = len(input_ids_orig)
            print('>>', tweet)
            print('>>', selected_text)
        
        sentiment_id = {
            'positive': tokenizer.token_to_id('positive'),
            'negative': tokenizer.token_to_id('negative'),
            'neutral': tokenizer.token_to_id('neutral')
        }
        
        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4
        
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets,
        }
    
    def __init__(self, tweet, sentiment, selected_text, tweet_id=None, name='roberta', max_len=192, **kwargs):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        ROBERTA_PATH = name
        self.tokenizer = tokenizers.SentencePieceBPETokenizer
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=f"{ROBERTA_PATH}/vocab.json",
            merges_file=f"{ROBERTA_PATH}/merges.txt",
            lowercase=True,
            add_prefix_space=True
        )
        self.max_len = max_len
        self.tweet_id = tweet_id
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        data = self.process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            'tweet_id': self.tweet_id[item],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


class XLNetLoader:
    @staticmethod
    def get_offsets(s, pattern, ignore=0):
        base = 0
        offset = []
        s = s.lower().replace('â', 'a')
        for idx, p in enumerate(pattern):
            if idx < ignore:
                offset.append((0, 0))
                continue
            start_idx, end_idx = AlbertLoader.find_index(s[base:], pattern, idx)
            offset.append((base + start_idx, base + end_idx))
            base += end_idx
        return offset
    
    @staticmethod
    def find_index(s, pattern, idx, roll=False):
        pattern[idx] = pattern[idx].lower().replace('▁', '')
        if pattern[idx] == '<unk>' and idx + 1 < len(pattern):
            start, end = AlbertLoader.find_index(s, pattern, idx + 1, roll=True)
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
    
    @staticmethod
    def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
        # handle special example
        tweet = tweet.replace('ï¿½', '').replace('¿½t', '').replace('ï', '').replace('``', '"') \
            .replace('^', '#').replace("''", '"')
        selected_text = selected_text.replace('ï¿½', '').replace('¿½t', '').replace('ï', '').replace('``', '"') \
            .replace('^', '#').replace("''", '"')
        
        raw_tweet = " " + " ".join(str(tweet).split())
        raw_selected_text = " " + " ".join(str(selected_text).split())
        
        tweet = raw_tweet.replace('-', '#')
        selected_text = raw_selected_text.replace('-', '#')
        
        question, text = sentiment, tweet
        encoding = tokenizer.encode_plus(question, text)
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        mask = [1] * len(token_type_ids)
        tweet_offsets = AlbertLoader.get_offsets(tweet, tokenizer.convert_ids_to_tokens(input_ids)[1:-1], ignore=1)
        tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]
        
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None
        
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        try:
            targets_start = target_idx[0]
            targets_end = target_idx[-1]
        except:
            targets_start = 2
            targets_end = len(input_ids) - 2
            print('>>', tweet)
            print('>>', selected_text)
        
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': raw_tweet,
            'orig_selected': raw_selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets,
        }
    
    def __init__(self, tweet, sentiment, selected_text, tweet_id=None, name='albert-xxlarge-v2', max_len=192):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = XLNetTokenizer.from_pretrained(name)
        self.max_len = max_len
        self.tweet_id = tweet_id
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        data = self.process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
            'tweet_id': self.tweet_id[item],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


class LinearHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        self.l0 = nn.Linear(d_model * layers_used, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, out, mask):
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits': start_logits, 'end_logits': end_logits}


class CNN(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2, div=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        self.cnn = nn.ModuleList([nn.Conv1d(d_model * layers_used // (div ** l),
                                            d_model * layers_used // (div ** (l + 1)), 5, padding=2)
                                  for l in range(num_layers)])
    
    def forward(self, out, mask=None):
        for cnn in self.cnn:
            out = cnn(out)
            out = F.leaky_relu(out)
        
        return out


class CNNHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2, div=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        self.cnn = CNN(d_model, layers_used=layers_used, num_layers=num_layers, div=div)
        self.l0 = nn.Linear(d_model * layers_used // (div ** num_layers), 2)
        for param in self.cnn.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.l0.weight)
    
    def forward(self, out, mask=None):
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)  # bs x SL x (768 * 2)
        out = self.drop_out(out)  # bs x SL x (768 * 2)
        out = out.permute(0, 2, 1)
        tar_sz = out.size()
        out = self.cnn(out)
        out = out.permute(0, 2, 1)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits': start_logits, 'end_logits': end_logits}


class StackCNNHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=4, div=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        self.cnn1 = CNN(d_model, layers_used=1, num_layers=num_layers, div=div)
        self.cnn2 = CNN(d_model, layers_used=1, num_layers=num_layers, div=div)
        self.l1 = nn.Linear(d_model * 1 // (div ** num_layers), 1)
        self.l2 = nn.Linear(d_model * 1 // (div ** num_layers), 1)
        for param in self.cnn1.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.cnn2.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.l1.weight)
        xavier_uniform_(self.l2.weight)
    
    def forward(self, out, mask=None):
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.stack(out)
        out = torch.mean(out, 0)
        out = self.drop_out(out)  # bs x SL x (768 * 2)
        out = out.permute(0, 2, 1)
        
        out1 = self.cnn1(out)
        out1 = out1.permute(0, 2, 1)
        start_logits = self.l1(out1)
        start_logits = start_logits.squeeze(-1)
        
        out2 = self.cnn2(out)
        out2 = out2.permute(0, 2, 1)
        end_logits = self.l2(out2)
        end_logits = end_logits.squeeze(-1)
        
        return {'start_logits': start_logits, 'end_logits': end_logits}


class LSTMHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.1)
        self.d = nn.Linear(d_model * layers_used, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, bidirectional=True)
        self.l0 = nn.Linear(d_model * 2, 2)
        for param in self.lstm.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.d.weight)
        xavier_uniform_(self.l0.weight)
    
    def forward(self, out, mask=None):
        mask = mask.ne(0)
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = self.drop_out(out)
        out = self.d(out)
        out = self.lstm(out.transpose(0, 1))[0].transpose(0, 1)
        
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits': start_logits, 'end_logits': end_logits}


class GRUHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.1)
        self.d = nn.Linear(d_model * layers_used, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, bidirectional=True)
        self.l0 = nn.Linear(d_model * 2, 2)
        for param in self.gru.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.d.weight)
        xavier_uniform_(self.l0.weight)
    
    def forward(self, out, mask=None):
        mask = mask.ne(0)
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = self.drop_out(out)
        out = self.d(out)
        out = self.gru(out.transpose(0, 1))[0].transpose(0, 1)
        
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits': start_logits, 'end_logits': end_logits}


class TransformerHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.1)
        self.d = nn.Linear(d_model * layers_used, d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048),
            num_layers=num_layers,
            norm=LayerNorm(d_model))
        self.l0 = nn.Linear(d_model, 2)
        for param in self.transformer.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.d.weight)
        xavier_uniform_(self.l0.weight)
    
    def forward(self, out, mask=None):
        mask = mask.ne(0)
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = self.drop_out(out)
        out = self.d(out)
        out = self.transformer(out.transpose(0, 1), src_key_padding_mask=~mask).transpose(0, 1)
        
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits': start_logits, 'end_logits': end_logits}


class MixHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2, div=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        
        self.d0 = nn.Linear(d_model * layers_used, d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048),
            num_layers=num_layers,
            norm=LayerNorm(d_model))
        
        self.d1 = nn.Linear(d_model * layers_used, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, bidirectional=True)
        
        self.cnn = CNN(d_model, layers_used=layers_used, num_layers=num_layers, div=div)
        
        self.l0 = nn.Linear(d_model * 3 + d_model * layers_used // (div ** num_layers) + d_model * num_layers, 2)
        
        for param in self.transformer.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.lstm.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.cnn.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.d0.weight)
        xavier_uniform_(self.d1.weight)
        xavier_uniform_(self.l0.weight)
    
    def forward(self, out, mask=None):
        mask = mask.ne(0)
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = self.drop_out(out)
        
        out1 = self.d0(out)
        out1 = self.transformer(out1.transpose(0, 1), src_key_padding_mask=~mask).transpose(0, 1)
        
        out2 = self.d1(out)
        out2 = self.lstm(out2.transpose(0, 1))[0].transpose(0, 1)
        
        out3 = out.permute(0, 2, 1)
        out3 = self.cnn(out3)
        out3 = out3.permute(0, 2, 1)
        
        out4 = out
        
        out = torch.cat((out1, out2, out3, out4), dim=-1)
        out = self.drop_out(out)
        
        logits = self.l0(out)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits': start_logits, 'end_logits': end_logits}


class SpanHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        self.l0 = nn.Linear(d_model * layers_used, 2)
        self.l1 = nn.Linear(d_model * layers_used, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
    
    def forward(self, out, mask):
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = self.drop_out(out)
        
        span = self.l1(out)
        
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return {'start_logits': start_logits, 'end_logits': end_logits, 'span': span}


class SpanCNNHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2, div=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        self.cnn = CNN(d_model, layers_used=layers_used, num_layers=num_layers, div=div)
        self.l0 = nn.Linear(d_model * layers_used // (div ** num_layers) + 2, 2)
        self.cnn1 = CNN(d_model, layers_used=layers_used, num_layers=num_layers, div=div)
        self.l1 = nn.Linear(d_model * layers_used // (div ** num_layers), 1)
        
        self.cnn2 = nn.Conv1d(1, 2, 5, padding=2)
        
        for param in self.cnn.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.l0.weight)
        for param in self.cnn1.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.cnn2.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.l1.weight)
    
    def forward(self, out, mask=None):
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)  # bs x SL x (768 * 2)
        out = self.drop_out(out)  # bs x SL x (768 * 2)
        out = out.permute(0, 2, 1)
        tar_sz = out.size()
        
        out1 = self.cnn1(out)
        out1 = out1.permute(0, 2, 1)
        span = self.l1(out1)
        
        span_used = self.cnn2(torch.sigmoid(span).permute(0, 2, 1)).view(tar_sz[0], 2, tar_sz[2]).permute(0, 2, 1)
        
        out0 = self.cnn(out)
        out0 = out0.permute(0, 2, 1)
        out0 = self.drop_out(out0)
        
        out0 = torch.cat([out0, span_used], dim=-1)
        logits = self.l0(out0)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return {'start_logits': start_logits, 'end_logits': end_logits, 'span': span}


class SpanMixHead(nn.Module):
    def __init__(self, d_model, layers_used, num_layers=2, div=2):
        super().__init__()
        self.d_model = d_model
        self.layers_used = layers_used
        self.drop_out = nn.Dropout(0.15)
        
        self.d0 = nn.Linear(d_model * layers_used // (div * num_layers), 1)
        
        self.d1 = nn.Linear(d_model * layers_used, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, bidirectional=True)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, bidirectional=True)
        self.cnn = CNN(d_model, layers_used=layers_used, num_layers=num_layers, div=div)
        self.cnn2 = CNN(d_model, layers_used=layers_used, num_layers=num_layers, div=div)
        self.cnn3 = nn.Conv1d(1, 2, 5, padding=2)
        
        self.l0 = nn.Linear(d_model * 4 + d_model * layers_used // (div ** num_layers) + d_model * layers_used + 2, 2)
        for param in self.lstm.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.gru.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.cnn.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.cnn2.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        for param in self.cnn3.parameters():
            if param.data.dim() > 1:
                xavier_uniform_(param.data)
        xavier_uniform_(self.d0.weight)
        xavier_uniform_(self.d1.weight)
        xavier_uniform_(self.l0.weight)
    
    def forward(self, out, mask=None):
        mask = mask.ne(0)
        out = [out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = self.drop_out(out)
        
        out0 = out.permute(0, 2, 1)
        tar_sz = out0.size()
        out0 = self.cnn2(out0)
        out0 = out0.permute(0, 2, 1)
        span = self.d0(out0)
        
        out2 = self.d1(out)
        out1 = self.gru(out2.transpose(0, 1))[0].transpose(0, 1)
        out2 = self.lstm(out2.transpose(0, 1))[0].transpose(0, 1)
        
        out3 = out.permute(0, 2, 1)
        out3 = self.cnn(out3)
        out3 = out3.permute(0, 2, 1)
        
        out4 = out
        
        span_used = self.cnn3(torch.sigmoid(span).permute(0, 2, 1)).view(tar_sz[0], 2, tar_sz[2]).permute(0, 2, 1)
        
        out = torch.cat((out1, out2, out3, out4), dim=-1)
        out = self.drop_out(out)
        out = torch.cat((out, span_used), dim=-1)
        
        logits = self.l0(out)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return {'start_logits': start_logits, 'end_logits': end_logits, 'span': span}


def train_fn(data_loader, model, optimizer, device, loss_fn=None, scheduler=None):
    model.train()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader), ncols=80)
    
    for bi, d in enumerate(tk0):
        tweet_id = d['tweet_id']
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]
        
        # Move ids, masks, and targets to gpu while setting as torch.long
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        
        # Reset gradients
        model.zero_grad()
        # Use ids, masks, and token types as input to the model
        # Predict logits for each of the input tokens for each batch
        output = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )  # (bs x SL), (bs x SL)
        outputs_start = output['start_logits']
        outputs_end = output['end_logits']
        # Calculate batch loss based on CrossEntropy
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, mask, output)
        # Calculate gradients based on loss
        loss.backward()
        # Adjust weights based on calculated gradients
        optimizer.step()
        # Update scheduler
        scheduler.step()
        
        # Apply softmax to the start and end logits
        # This squeezes each of the logits in a sequence to a value between 0 and 1, while ensuring that they sum to 1
        # This is similar to the characteristics of "probabilities"
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        
        # Calculate the jaccard score based on the predictions for this batch
        jaccard_scores = []
        for px, (tweet_id, tweet) in enumerate(zip(tweet_id, orig_tweet)):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,  # Full text of the px'th tweet in the batch
                target_string=selected_tweet,
                # Span containing the specified sentiment for the px'th tweet in the batch
                sentiment_val=tweet_sentiment,  # Sentiment of the px'th tweet in the batch
                idx_start=outputs_start[px, :],  # Predicted start index for the px'th tweet in the batch
                idx_end=outputs_end[px, :],  # Predicted end index for the px'th tweet in the batch
                offsets=offsets[px]  # Offsets for each of the tokens for the px'th tweet in the batch
            )
            # if 'new' in tweet_id:
            #     continue
            jaccard_scores.append(jaccard_score)
        # Update the jaccard score and loss
        # For details, refer to `AverageMeter` in https://www.kaggle.com/abhishek/utils
        avg = np.mean(jaccard_scores) if len(jaccard_scores) else 0
        jaccards.update(avg, len(jaccard_scores))
        losses.update(loss.item(), ids.size(0))
        # Print the average loss and jaccard score at the end of each batch
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
        # if bi % 100 == 0:
        #     print(bi, losses.avg, jaccards.avg)


def calculate_jaccard_score(
        original_tweet,
        target_string,
        sentiment_val,
        idx_start,
        idx_end,
        offsets,
        verbose=False):
    output_start = idx_start
    output_end = idx_end
    arg_start = np.argsort(output_start)
    arg_end = np.argsort(output_end)
    idx_start = arg_start[-1]
    idx_end = arg_end[-1]
    if idx_end < idx_start:
        start = idx_start
        end = idx_start
        # second_start = arg_start[-2]
        # second_end = arg_end[-2]
        # prob = 0
        # # first + second
        # if idx_start < second_end and prob < output_start[idx_start] * output_end[second_end]:
        #     start = idx_start
        #     end = second_end
        #     prob = output_start[idx_start] * output_end[second_end]
        # # second + first
        # if second_start < idx_end and prob < output_start[second_start] * output_end[idx_end]:
        #     start = second_start
        #     end = idx_end
        #     prob = output_start[second_start] * output_end[idx_end]
        # # second + second
        # if second_start < second_end and prob < output_start[second_start] * output_end[second_end]:
        #     start = second_start
        #     end = second_end
        #     prob = output_start[second_start] * output_end[second_end]
        idx_start = start
        idx_end = end
    
    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "
    
    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet
    
    jac = utils.jaccard(target_string.strip(), filtered_output.strip())
    
    # filtered_output = filtered_output.strip()
    # # 1
    # filtered_output = filtered_output
    #
    # # 2
    # if filtered_output[0] == '"':
    #     filtered_output = filtered_output[1:]
    # if filtered_output[-1] == '"':
    #     filtered_output = filtered_output[:-1]
    #
    # # 3
    # if filtered_output[0] != '"':
    #     filtered_output = '"' + filtered_output
    # if filtered_output[-1] != '"':
    #     filtered_output = filtered_output + '"'
    
    return jac, filtered_output


def eval_fn(data_loader, model, device, loss_fn=None):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), ncols=80)
        for bi, d in enumerate(tk0):
            tweet_id = d['tweet_id']
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()
            
            # Move ids, masks, and targets to gpu while setting as torch.long
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            
            # Move tensors to GPU for faster matrix calculations
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            
            # Predict logits for start and end indexes
            output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start = output['start_logits']
            outputs_end = output['end_logits']
            # Calculate loss for the batch
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, mask, output)
            # Apply softmax to the predicted logits for the start and end indexes
            # This converts the "logits" to "probability-like" scores
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            # Calculate jaccard scores for each tweet in the batch
            jaccard_scores = []
            for px, (tweet_id, tweet) in enumerate(zip(tweet_id, orig_tweet)):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,  # Full text of the px'th tweet in the batch
                    target_string=selected_tweet,
                    # Span containing the specified sentiment for the px'th tweet in the batch
                    sentiment_val=tweet_sentiment,  # Sentiment of the px'th tweet in the batch
                    idx_start=outputs_start[px, :],  # Predicted start index for the px'th tweet in the batch
                    idx_end=outputs_end[px, :],  # Predicted end index for the px'th tweet in the batch
                    offsets=offsets[px]  # Offsets for each of the tokens for the px'th tweet in the batch
                )
                # if 'new' in tweet_id:
                #     continue
                jaccard_scores.append(jaccard_score)
            # Update the jaccard score and loss
            # For details, refer to `AverageMeter` in https://www.kaggle.com/abhishek/utils
            avg = np.mean(jaccard_scores) if len(jaccard_scores) else 0
            jaccards.update(avg, len(jaccard_scores))
            losses.update(loss.item(), ids.size(0))
            # Print the average loss and jaccard score at the end of each batch
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
            # if bi % 100 == 0:
            #     print(bi, losses.avg, jaccards.avg)
    
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg


def build_data(path, builder, train_batch_size, val_batch_size, name):
    df_test = pd.read_csv("data/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values
    test_dataset = builder(df_test.text.values, df_test.sentiment.values, df_test.selected_text.values,
                           tweet_id=df_test.textID.values, max_len=192, name=name)
    
    # Instantiate DataLoader with `test_dataset`
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=val_batch_size,
        num_workers=1
    )
    return data_loader


def run(fold, path, data, model, batch_size, epochs, loss_fn, save_path, lr=3e-5, scheduler_fn=None, name='roberta',
        pretrained=False):
    dfx = pd.read_csv(path)
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    train_dataset = data(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        tweet_id=df_train.textID.values,
        name=name
    )
    
    # Instantiate DataLoader with `train_dataset`
    # This is a generator that yields the dataset in batches
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4
    )
    
    # Instantiate TweetDataset with validation data
    valid_dataset = data(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values,
        tweet_id=df_valid.textID.values,
        name=name
    )
    
    # Instantiate DataLoader with `valid_dataset`
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=2
    )
    
    device = torch.device("cuda")
    model = copy.deepcopy(model)
    model.to(device)
    if pretrained:
        try:
            model.load_state_dict(torch.load(save_path + str(fold) + '.bin'))
            print('load pretrained')
        except:
            print('load fail')
    num_train_steps = int(len(df_train) / batch_size * epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    # Instantiate AdamW optimizer with our two sets of parameters, and a learning rate of 3e-5
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = scheduler_fn(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    es = utils.EarlyStopping(patience=2, mode="max")
    print(f"Training is Starting for fold={fold}")
    
    # jaccard = eval_fn(valid_data_loader, model, device, loss_fn=loss_fn)
    # print(f"Jaccard Score = {jaccard}")
    
    for epoch in range(epochs):
        train_fn(train_data_loader, model, optimizer, device, loss_fn=loss_fn, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device, loss_fn=loss_fn)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=save_path + str(fold) + '.bin')
        if es.early_stop:
            print("Early stopping")
            break


def train_folds(folds, path, data, model, batch_size, epochs, loss_fn, save_path, lr=3e-5, scheduler_fn=None,
                name='roberta', pretrained=False):
    for fold in folds:
        run(fold, path, data, model, batch_size, epochs, loss_fn, save_path, lr=lr, scheduler_fn=scheduler_fn,
            name=name,
            pretrained=pretrained)


def test(model, data_loader, SAVE_HEAD, MODE):
    device = torch.device("cuda")
    
    # Load each of the five trained models and move to GPU
    model1 = model
    model1.to(device)
    model1.load_state_dict(torch.load(SAVE_HEAD + '0.bin'))
    model1.eval()
    
    model2 = copy.deepcopy(model)
    model2.to(device)
    model2.load_state_dict(torch.load(SAVE_HEAD + '1.bin'))
    model2.eval()
    
    model3 = copy.deepcopy(model)
    model3.to(device)
    model3.load_state_dict(torch.load(SAVE_HEAD + '2.bin'))
    model3.eval()
    
    model4 = copy.deepcopy(model)
    model4.to(device)
    model4.load_state_dict(torch.load(SAVE_HEAD + '3.bin'))
    model4.eval()
    
    model5 = copy.deepcopy(model)
    model5.to(device)
    model5.load_state_dict(torch.load(SAVE_HEAD + '4.bin'))
    model5.eval()
    
    final_output = []
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), ncols=80)
        # Predict the span containing the sentiment for each batch
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()
            
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            
            # Predict start and end logits for each of the five models
            output1 = model1(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start1 = output1['start_logits']
            outputs_end1 = output1['end_logits']
            
            output2 = model2(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start2 = output2['start_logits']
            outputs_end2 = output2['end_logits']
            
            output3 = model3(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start3 = output3['start_logits']
            outputs_end3 = output3['end_logits']
            
            output4 = model4(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start4 = output4['start_logits']
            outputs_end4 = output4['end_logits']
            
            output5 = model5(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start5 = output5['start_logits']
            outputs_end5 = output5['end_logits']
            
            # Get the average start and end logits across the five models and use these as predictions
            # This is a form of "ensembling"
            outputs_start = (
                                    outputs_start1
                                    + outputs_start2
                                    + outputs_start3
                                    + outputs_start4
                                    + outputs_start5
                            ) / 5
            outputs_end = (
                                  outputs_end1
                                  + outputs_end2
                                  + outputs_end3
                                  + outputs_end4
                                  + outputs_end5
                          ) / 5
            
            # Apply softmax to the predicted start and end logits
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            
            # Convert the start and end scores to actual predicted spans (in string form)
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                _, output_sentence = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=outputs_start[px, :],
                    idx_end=outputs_end[px, :],
                    offsets=offsets[px]
                )
                final_output.append(output_sentence)
    
    # post-process trick:
    # Note: This trick comes from: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942
    # When the LB resets, this trick won't help
    def post_process(selected):
        return selected
        # return " ".join(set(selected.lower().split()))
    
    sample = pd.read_csv("data/sample_submission.csv")
    sample.loc[:, 'selected_text'] = final_output
    sample.selected_text = sample.selected_text.map(post_process)
    sample.to_csv(MODE + "submission.csv", index=False)
    
    sample.head()


def test_folds(folds, model, data_loader, SAVE_HEAD, MODE):
    device = torch.device("cuda")
    
    # Load each of the five trained models and move to GPU
    folds_num = len(folds)
    ensemble = nn.ModuleList([])
    for i in range(folds[-1] + 1):
        _model = copy.deepcopy(model)
        _model.to(device)
        _model.load_state_dict(torch.load(SAVE_HEAD + str(i) + '.bin'))
        _model.eval()
        ensemble.append(_model)
    
    final_output = []
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), ncols=80)
        # Predict the span containing the sentiment for each batch
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()
            
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            
            # Predict start and end logits for each of the five models
            output = []
            output_start_list = []
            output_end_list = []
            for i, m in enumerate(ensemble):
                _output = m(ids=ids, mask=mask, token_type_ids=token_type_ids)
                _outputs_start = _output['start_logits']
                _outputs_end = _output['end_logits']
                output.append(_output)
                output_start_list.append(_outputs_start)
                output_end_list.append(_outputs_end)
            
            # Get the average start and end logits across the five models and use these as predictions
            # This is a form of "ensembling"
            outputs_start = output_start_list[0]
            outputs_end = output_end_list[0]
            for i in range(1, len(output_start_list)):
                outputs_start = outputs_start + output_start_list[i]
                outputs_end = outputs_end + output_end_list[i]
            outputs_start = outputs_start / folds_num
            outputs_end = outputs_end / folds_num
            
            # Apply softmax to the predicted start and end logits
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            
            # Convert the start and end scores to actual predicted spans (in string form)
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                _, output_sentence = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=outputs_start[px, :],
                    idx_end=outputs_end[px, :],
                    offsets=offsets[px]
                )
                final_output.append(output_sentence)
    
    # post-process trick:
    # Note: This trick comes from: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942
    # When the LB resets, this trick won't help
    def post_process(selected):
        # return " ".join(set(selected.lower().split()))
        return selected
    
    sample = pd.read_csv("data/sample_submission.csv")
    sample.loc[:, 'selected_text'] = final_output
    sample.selected_text = sample.selected_text.map(post_process)
    sample.to_csv(MODE + "submission.csv", index=False)
    
    sample.head()


def get_loss_fn(ce=1., dst=0., span=0., jcd=0., lvs=0., jel=0., ksl=0., **kwargs):
    def f(start_logits, end_logits, start_positions, end_positions, mask=None, output=None):
        c_loss = ce_loss(start_logits, end_logits, start_positions, end_positions)
        loss = ce * c_loss
        if dst != 0:
            loss = loss + dst * distance_loss(start_logits, end_logits, start_positions, end_positions, mask=mask)
        if span != 0:
            loss = loss + span * span_loss(start_positions, end_positions, mask, output, jcd, lvs)
        if jel != 0:
            loss = loss + jel * jel_loss(start_logits, end_logits, start_positions, end_positions)
        if ksl != 0:
            loss = loss + ksl * (KSLoss(start_logits, start_positions) + KSLoss(end_logits, end_positions)) / 2
        
        return loss
    
    return f


def ce_loss(start_logits, end_logits, start_positions, end_positions, mask=None):
    """
        Return the sum of the cross entropy losses for both the start and end logits
        """
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def _get_score(start, end, max_len, mask=None):
    start_score = []
    start = start.item()
    end = end.item()
    for i in range(max_len):
        if mask is not None and i < len(mask) and mask[i] == 0:
            start_score.append(0)
        else:
            start_score.append(min(end + 1 - start, max(end + 1 - i, 0)) / max(end + 1 - start, end + 1 - i))
    end_score = []
    for j in range(max_len):
        if mask is not None and j < len(mask) and mask[j] == 0:
            end_score.append(0)
        else:
            end_score.append(min(end + 1 - start, max(j + 1 - start, 0)) / max(end + 1 - start, j + 1 - start))
    return start_score, end_score


def jcd_score(start_positions, end_positions, mask=None):
    bc_size = start_positions.size(0)
    start_score = []
    end_score = []
    for i in range(bc_size):
        start, end = _get_score(start_positions[i], end_positions[i], 200, mask[i])
        start_score.append(start)
        end_score.append(end)
    return start_score, end_score


def distance_loss(start_logits, end_logits, start_positions, end_positions, mask=None, temperature=0):
    start_score, end_score = jcd_score(start_positions, end_positions, mask)
    start_score = torch.tensor(start_score).to(start_logits.device) ** 2
    end_score = torch.tensor(end_score).to(end_logits.device) ** 2
    start_loss = -(F.log_softmax(start_logits, dim=-1) * start_score[:, :start_logits.size(1)]).mean()
    end_loss = -(F.log_softmax(end_logits, dim=-1) * end_score[:, :end_logits.size(1)]).mean()
    total_loss = (start_loss + end_loss) / 2
    return total_loss


def span_loss(start_positions, end_positions, mask, output, jcd=0., lvs=0.):
    span = output['span']
    bc_size = span.size(0)
    target = torch.zeros_like(span)
    for b in range(bc_size):
        target[b][start_positions[b]:end_positions[b]] = 1.
    target = target.to(span.device)
    loss = (1 - jcd - lvs) * F.binary_cross_entropy(torch.sigmoid(span), target)
    if jcd != 0:
        loss = loss + jcd * jaccard_loss(torch.sigmoid(span), target).mean()
    if lvs != 0:
        loss = loss + lvs * lovasz_hinge_flat(span.view(-1), target.view(-1)).mean()
    
    return loss


def jaccard_loss(pred, target, smooth=1e-10):
    I = (pred * target).sum(axis=1, keepdim=True)
    P = pred.sum(axis=1, keepdim=True)
    T = target.sum(axis=1, keepdim=True)
    loss = 1 - ((I + smooth) / (P + T - I + smooth))
    return loss


def jel_loss(start_logits, end_logits, start_positions, end_positions):
    # from https://www.kaggle.com/koza4ukdmitrij/jaccard-expectation-loss/comments
    indexes = torch.arange(start_logits.size()[1]).unsqueeze(0).cuda()
    
    start_pred = torch.sum(F.softmax(start_logits, dim=-1) * indexes, dim=1)
    end_pred = torch.sum(F.softmax(end_logits, dim=-1) * indexes, dim=1)
    
    len_true = end_positions - start_positions + 1
    intersection = len_true - F.relu(start_pred - start_positions) - F.relu(end_positions - end_pred)
    union = len_true + F.relu(start_positions - start_pred) + F.relu(end_pred - end_positions)
    
    jel = 1 - intersection / union
    jel = torch.mean(jel)
    return jel


def KSLoss(preds, label):
    bc_size = preds.size(0)
    target = torch.zeros_like(preds)
    for b in range(bc_size):
        target[b][label[b]] = 1.
    target = target.to(preds.device)
    
    pred_cdf = torch.cumsum(torch.softmax(preds, dim=1), dim=1)
    target_cdf = torch.cumsum(target, dim=1)
    error = (target_cdf - pred_cdf) ** 2
    return torch.mean(error)


def collect(model):
    param_num = 0
    escape = 0
    for name, param in model.named_parameters():
        s_ = 1
        for ax_ in param.size():
            s_ *= ax_
        param_num += s_
    return param_num


def decode_exp(exp):
    exp = exp.split(',')
    m = {}
    for e in exp:
        e = e.split('=')
        if len(e) == 2:
            m[e[0]] = float(e[1])
    return m
