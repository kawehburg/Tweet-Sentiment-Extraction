import numpy as np
import pandas as pd
import os
import warnings
import random
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
import copy


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed = 42
seed_everything(seed)
data_save = {}

bptokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file='./roberta/vocab.json',
    merges_file='./roberta/merges.txt',
    lowercase=True,
    add_prefix_space=True)

setlist = []
fw = open("testfin.txt", "w", encoding='utf-8')


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='./roberta/vocab.json',
            merges_file='./roberta/merges.txt',
            lowercase=True,
            add_prefix_space=True)
    
    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        
        ids, masks, tweet, offsets, origin_str = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        data['origin_str'] = origin_str
        
        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets, origin_str)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
        
        return data
    
    def __len__(self):
        return len(self.df)
    
    def get_input_data(self, row):
        rowid = row.textID
        if row.textID in ["af3fed7fc3", "95e12b1cb1", "a54d3c2825", "24f090ea3d", "8a8c28f5ba"]:
            pass
        
        origin_str = row.text.lower()
        
        tweet = " " + " ".join(row.text.lower().split())
        
        encoding = self.tokenizer.encode(tweet)
        
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
        
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
        
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        
        return ids, masks, tweet, offsets, origin_str
    
    def get_target_idx(self, row, tweet, offsets, origin_str):
        selected_text = " " + " ".join(row.selected_text.lower().split())
        
        if row.textID in ["af3fed7fc3", "95e12b1cb1", "a54d3c2825", "24f090ea3d", "8a8c28f5ba"]:
            pass
        
        # 找到原来有空格但是现在没有的地方
        len_ori = len(origin_str)
        len_twi = len(tweet)
        spc_targets = [0] * len(tweet)
        i = 0
        j = 0
        while True:
            if j >= len_twi:
                break
            a = origin_str[i]
            b = tweet[j]
            if a == b:
                i += 1
                j += 1
            else:
                if origin_str[i] == " ":
                    spc_targets[j] += 1
                    i += 1
                
                if tweet[j] == " ":
                    j += 1
                
                if origin_str[i] == "\t" or origin_str[i] == "\xa0":
                    i += 1
        
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
        
        spc_cnt = 0
        for i in range(len(tweet)):
            if char_targets[i] == 1:
                break
            
            if spc_targets[i] != 0:
                spc_cnt += spc_targets[i]
        
        # 计数的，你可以看看slimfin.txt
        # if spc_cnt > 0:
        #    setlist.append(row.textID + " | " + str(spc_cnt) + " | " + origin_str + " | " + selected_text)
        #    fw.write(row.textID + " | " + str(spc_cnt) + " | " + origin_str + " | " + selected_text)
        #    fw.write("\n")
        #    fw.flush()
        #    print("12313")
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        
        seltextget = get_selected_text(tweet, start_idx, end_idx, offsets, origin_str)
        
        return start_idx, end_idx


def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    
    return dataloaders_dict


def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    return loader


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        
        config = RobertaConfig.from_pretrained(
            './roberta/config.json', output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(
            './roberta/pytorch_model.bin', config=config)
        self.dropout = nn.Dropout(0.15)
        self.cnn1 = nn.Sequential(
            torch.nn.Conv1d(config.hidden_size, 128, 2),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU()
        )
        self.cnn1_1 = nn.Sequential(
            torch.nn.Conv1d(128, 64, 2),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU()
        )
        self.cnn2 = nn.Sequential(
            torch.nn.Conv1d(config.hidden_size, 128, 2),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU()
        )
        self.cnn2_1 = nn.Sequential(
            torch.nn.Conv1d(128, 64, 2),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.normal_(self.fc2.bias, 0)
    
    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)
        
        # x = torch.stack([hs[-1], hs[-2], hs[-3]])
        # x = torch.mean(x, 0)
        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x1 = self.cnn1_1(x1)
        x2 = self.cnn2_1(x2)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        start_logits = self.fc1(x1)
        end_logits = self.fc2(x2)
        # start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def get_selected_text(text, start_idx, end_idx, offsets, origin_text):
    selected_text = ""
    reg_rule = re.compile(r"[ï¿½â]")  # 处理奇葩中文字符，暴力
    reg_all = reg_rule.findall(text)
    
    start_index = -1
    end_index = -1
    if len(reg_all) != 0:
        tok_new = bptokenizer.encode(text)
        new_offset = []
        for _ in range(4):
            new_offset.append(tuple([0, 0]))
        speciallist = ["Ã¯", "Â", "Â½", "Ã¢", "Â´"]
        offset_calc = 0
        for i, tok in enumerate(tok_new.tokens):
            if tok in speciallist:
                if tok == "Ã¯":
                    tmpl = list(tok_new.offsets[i])
                    tmpl[0] = tmpl[0] - offset_calc
                    tmpl[1] = tmpl[1] - offset_calc - 1
                    offset_calc += 1
                    new_offset.append(tuple(tmpl))
                
                if tok == "Â":
                    if tok_new.tokens[i - 1] == "Ã¯":
                        new_offset.append(new_offset[-1])
                        offset_calc += 1
                
                if tok == "Â½":
                    tmpl = list(tok_new.offsets[i])
                    tmpl[0] = tmpl[0] - offset_calc
                    tmpl[1] = tmpl[1] - offset_calc - 1
                    offset_calc += 1
                    new_offset.append(tuple(tmpl))
                
                if tok == "Ã¢":
                    get_offset = list(tok_new.offsets[i])
                    get_offset[0] = get_offset[0] - offset_calc
                    get_offset[1] = get_offset[1] - offset_calc - 1
                    new_offset.append(tuple(get_offset))
                    offset_calc += 1
                
                if tok == "Â´":
                    get_offset = list(tok_new.offsets[i])
                    get_offset[0] = get_offset[0] - offset_calc
                    get_offset[1] = get_offset[1] - offset_calc - 1
                    new_offset.append(tuple(get_offset))
                    offset_calc += 1
            
            else:
                if offset_calc != 0:
                    get_offset = list(tok_new.offsets[i])
                    get_offset[0] = get_offset[0] - offset_calc
                    get_offset[1] = get_offset[1] - offset_calc
                    new_offset.append(tuple(get_offset))
                else:
                    new_offset.append(tok_new.offsets[i])
        
        if len(new_offset) < 96:
            for _ in range(96 - len(new_offset)):
                new_offset.append(tuple([0, 0]))
        
        asa = len(text)
        for ix in range(start_idx, end_idx + 1):
            if ix > start_idx:
                if new_offset[ix] == new_offset[ix - 1]:
                    continue
                else:
                    selected_text += text[new_offset[ix][0]: new_offset[ix][1]]
            else:
                selected_text += text[new_offset[ix][0]: new_offset[ix][1]]
            
            if (ix + 1) < len(new_offset) and new_offset[ix][1] < new_offset[ix + 1][0]:
                selected_text += " "
    
    else:
        for ix in range(start_idx, end_idx + 1):
            if start_index == -1:
                start_index = offsets[ix][0]
            selected_text += text[offsets[ix][0]: offsets[ix][1]]
            if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                selected_text += " "
    
    space_cnt = 0
    space_flg = False
    for i in range(len(origin_text)):
        if i > start_index:
            break
        if origin_text[i] == " ":
            if space_flg == False:
                space_flg = True
            else:
                space_cnt += 1
        else:
            space_flg = False
    
    if space_cnt > 1:
        if start_index - space_cnt > 0:
            st_idx = start_index - space_cnt
            selen = len(selected_text)
            ed_idx = selen + st_idx
            selected_text = origin_text[st_idx:ed_idx]
        # print(123)
    
    if selected_text == "":
        # print("123")
        pass
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets, origin_text):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets, origin_text)
    
    true = get_selected_text(text, start_idx, end_idx, offsets, origin_text)
    
    return jaccard(true, pred)


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.cuda()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            epoch_loss = 0.0
            epoch_jaccard = 0.0
            
            for data in (dataloaders_dict[phase]):
                ids = data['ids'].cuda()
                masks = data['masks'].cuda()
                tweet = data['tweet']
                offsets = data['offsets'].numpy()
                start_idx = data['start_idx'].cuda()
                end_idx = data['end_idx'].cuda()
                origin_text = data["origin_str"]
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    start_logits, end_logits = model(ids, masks)
                    
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item() * len(ids)
                    
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                    
                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i],
                            end_logits[i],
                            offsets[i],
                            origin_text[i])
                        epoch_jaccard += jaccard_score
            
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)
            
            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))
    
    torch.save(model.state_dict(), filename)


# 补1-2叹号比4个消掉好一点点cv
import re


def post_process(s):
    text = s
    try:
        a = re.findall('[^A-Za-z0-9]', s)
        b = re.sub('[^A-Za-z0-9]+', '', s)
        
        if a.count('!') >= 4:
            text = b + '! ' + b + '!! '
        else:
            text = s
        return text
    except:
        return text


if __name__ == '__main__':
    num_epochs = 3
    batch_size = 32
    skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=seed)
    
    train_df = pd.read_csv('./data/train.csv')
    train_df['text'] = train_df['text'].astype(str)
    train_df['selected_text'] = train_df['selected_text'].astype(str)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
        print(f'Fold: {fold}')
        
        model = TweetModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
        criterion = loss_fn
        dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)
        
        train_model(
            model,
            dataloaders_dict,
            criterion,
            optimizer,
            num_epochs,
            f'saved/roberta_fold{fold}.pth'
        )
    
    test_df = pd.read_csv('./data/test.csv')
    test_df['text'] = test_df['text'].astype(str)
    test_loader = get_test_loader(test_df)
    predictions = []
    models = []
    for fold in range(skf.n_splits):
        model = TweetModel()
        model.cuda()
        model.load_state_dict(torch.load(f'saved/roberta_fold{fold + 1}.pth'))
        model.eval()
        models.append(model)
    
    for data in test_loader:
        ids = data['ids'].cuda()
        masks = data['masks'].cuda()
        tweet = data['tweet']
        offsets = data['offsets'].numpy()
        origin_text = data["origin_str"]
        
        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())
        
        start_logits = np.mean(start_logits, axis=0)
        end_logits = np.mean(end_logits, axis=0)
        for i in range(len(ids)):
            start_pred = np.argmax(start_logits[i])
            end_pred = np.argmax(end_logits[i])
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i], origin_text[i])
            predictions.append(pred)
    
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df['selected_text'] = predictions
    sub_df.to_csv('submission_ori.csv', index=False)
    
    # sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: post_process(x) if len(x.split()) == 1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)
    # happy 的一定存在，缺了补
    for i, xp in enumerate(sub_df['selected_text']):
        if 'happy' in test_df['text'].iloc[i]:
            if 'happy' not in xp:
                xp = xp + " " + 'happy'
                sub_df.loc[i, 'selected_text'] = xp
    
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()

'''
    speciallist = ["Ã¯", "Â", "Â½"]
        offset_calc = 0
        if "ï¿½" in tweet:
            encoding = self.tokenizer.encode(tweet)
            for i, tok in enumerate(encoding.tokens):
                if tok in speciallist:
                    if tok == "Ã¯":
                        tmpl = list(encoding.offsets[i])
                        tmpl[-1] = tmpl[-1] - offset_calc-1
                        encoding.offsets[i] = tuple(tmpl)

                    if tok == "Â":
                        if encoding.tokens[i-1] == "Ã¯":
                            encoding.offsets[i] = encoding.offsets[i-1]
                else:
                    if offset_calc != 0:
                        encoding.offsets[i] = (encoding.offsets[i][0]-offset_calc , encoding.offsets[1]-offset_calc)



        else:


    train_df = pd.read_csv('./train.csv')
    train_df['text'] = train_df['text'].astype(str)
    train_df['selected_text'] = train_df['selected_text'].astype(str)    

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1): 
        print(f'Fold: {fold}')

        model = TweetModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
        criterion = loss_fn    
        dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

        train_model(
            model, 
            dataloaders_dict,
            criterion, 
            optimizer, 
            num_epochs,
            f'roberta_fold{fold}.pth'
            )
            '''
