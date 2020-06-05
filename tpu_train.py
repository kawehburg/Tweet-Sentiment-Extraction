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
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import copy

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")


def tpu_train_fn(data_loader, model, optimizer, device, num_batches, scheduler=None, loss_fn=None):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for bi, d in enumerate(tk0):
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
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        
        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        scheduler.step()
        tk0.set_postfix(loss=loss.item())



def tpu_eval_fn(data_loader, model, device, loss_fn=None):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].cpu().numpy()
            
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            
            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)
            
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=loss.item())
    
    return jaccards.avg


def tpu_run(fold, path, data, model, batch_size, epochs, loss_fn, save_path, lr=3e-5, scheduler_fn=None, name='roberta',
            pretrained=False):
    model = copy.deepcopy(model)
    device = xm.xla_device(fold + 1)
    model = model.to(device)
    
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
    
    num_train_steps = int(len(df_train) / batch_size * epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight"
    ]
    optimizer_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay
                )
            ],
            'weight_decay': 0.001
        },
        {
            'params': [
                p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay
                )
            ],
            'weight_decay': 0.0
        },
    ]
    num_train_steps = int(
        len(df_train) / batch_size * epochs
    )
    optimizer = AdamW(
        optimizer_parameters,
        lr=lr
    )
    scheduler = scheduler_fn(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    
    best_jac = 0
    es = utils.EarlyStopping(patience=2, mode="max")
    num_batches = int(len(df_train) / batch_size)
    
    for epoch in range(epochs):
        tpu_train_fn(
            train_data_loader,
            model,
            optimizer,
            device,
            num_batches,
            scheduler,
            loss_fn=loss_fn
        )
        
        jac = tpu_eval_fn(
            valid_data_loader,
            model,
            device,
            loss_fn=loss_fn
        )
        print(f'Epoch={epoch}, Fold={fold}, Jaccard={jac}')
        if jac > best_jac:
            xm.save(model.state_dict(), f"{save_path}{fold}.bin")
            best_jac = jac

def train_folds(folds, path, data, model, batch_size, epochs, loss_fn, save_path, lr=3e-5, scheduler_fn=None,
                name='roberta', pretrained=False):
    Parallel(n_jobs=len(folds), backend="threading")(delayed(tpu_run)(i, path, data, model, batch_size, epochs, loss_fn, save_path, lr=lr, scheduler_fn=scheduler_fn,
            name=name, pretrained=pretrained) for i in folds)

        
Parallel(n_jobs=8, backend="threading")(delayed(run)(i) for i in range(8))
