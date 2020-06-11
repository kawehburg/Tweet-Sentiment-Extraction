from tools.master import run, test, seed_everything, get_loss_fn, build_data, Model, collect
from tools.master import test_folds, train_folds, calculate_jaccard_score
from tools.master import BERTModel, ELECTRAModel, RoBERTaModel, Albert, Embedding, XLNet
from tools.master import BERTLoader, RoBERTaLoader, AlbertLoader, XLNetLoader
from tools.master import LinearHead, CNNHead, TransformerHead, LSTMHead, GRUHead, MixHead, SpanHead, SpanCNNHead, \
    SpanMixHead
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import os
import copy
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm.autonotebook import tqdm

model_list = {'bert': BERTModel, 'electra': ELECTRAModel, 'roberta': RoBERTaModel,
              'albert': Albert, 'embedding': Embedding, 'xlnet': XLNet}
pretrained_list = {'bert': ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased'],
                   'electra': ['google/electra-base-discriminator', 'google/electra-large-discriminator'],
                   'roberta': ['roberta'],
                   'albert': ['albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2'],
                   'embedding': ['embedding'],
                   'xlnet': ['xlnet-base-cased', 'xlnet-large-cased']}
config = {'bert-base-uncased': 768, 'bert-large-uncased': 1024, 'bert-base-cased': 768,
          'google/electra-base-discriminator': 768, 'google/electra-large-discriminator': 1024,
          'roberta': 768,
          'albert-base-v2': 768, 'albert-large-v2': 1024, 'albert-xlarge-v2': 2048, 'albert-xxlarge-v2': 4096,
          'embedding': 256,
          'xlnet-base-cased': 768, 'xlnet-large-cased': 1024
          }

data_list = {'bert': BERTLoader, 'electra': BERTLoader, 'roberta': RoBERTaLoader,
             'albert': AlbertLoader, 'embedding': BERTLoader, 'xlnet': XLNetLoader}

head_list = {'linear': LinearHead, 'cnn': CNNHead, 'transformer': TransformerHead, 'lstm': LSTMHead, 'gru': GRUHead,
             'mix': MixHead, 'span_linear': SpanHead, 'span_cnn': SpanCNNHead, 'span_mix': SpanMixHead}
schedule_list = {'linear_warmup': get_linear_schedule_with_warmup, 'cosine_warmup': get_cosine_schedule_with_warmup}


def build_model(folds, data_name, base_model, name, head, layers_used, num_layers, batch_size):
    d_model = config[name]
    save_path = f'saved/{data_name}_{base_model}_{d_model}_{head}_'
    data = data_list[base_model]
    test_data_loader = build_data("data/test.csv", data, batch_size, batch_size, name)
    base_model = model_list[base_model](name=name)
    head = head_list[head](d_model, layers_used, num_layers=num_layers)
    model = Model(base_model, head)
    print(save_path, 'param num =', collect(model))
    folds_num = len(folds)
    ensemble = nn.ModuleList([])
    device = torch.device("cuda")
    for i in folds:
        _model = copy.deepcopy(model)
        _model.to(device)
        _model.load_state_dict(torch.load(save_path + str(i) + '.bin'))
        _model.eval()
        ensemble.append(_model)
    return ensemble, test_data_loader, folds_num


def ext(var, idx):
    return [item[idx] for item in var]


def point_ensemble(ensemble_start, ensemble_end, ensemble_offset, max_len=192):
    start_prob = np.zeros(max_len)
    end_prob = np.zeros(max_len)
    for i in range(len(ensemble_offset)):
        for j in range(len(ensemble_start)):
            start_prob[ensemble_offset[i][j][0]] += ensemble_start[i][j]
            end_prob[ensemble_offset[i][j][1]] += ensemble_end[i][j]
    start_prob = start_prob / start_prob.sum()
    end_prob = end_prob / end_prob.sum()
    return start_prob, end_prob


def test_ensemble(models, data_loaders, folds_nums, result_path):
    device = torch.device("cuda")
    final_output = []
    with torch.no_grad():
        tk0 = tqdm(zip(data_loaders), total=len(data_loaders[0]), ncols=80)
        # Predict the span containing the sentiment for each batch
        for bi, data_loader in enumerate(tk0):
            ensemble_start = []
            ensemble_end = []
            ensemble_offset = []
            for idx, d in enumerate(data_loader):
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                sentiment = d["sentiment"]
                orig_tweet = d["orig_tweet"]
                offsets = d["offsets"].numpy()
                
                ids = ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                
                output = []
                output_start_list = []
                output_end_list = []
                for i, m in enumerate(models[idx]):
                    _output = m(ids=ids, mask=mask, token_type_ids=token_type_ids)
                    _outputs_start = _output['start_logits']
                    _outputs_end = _output['end_logits']
                    output.append(_output)
                    output_start_list.append(_outputs_start)
                    output_end_list.append(_outputs_end)
                outputs_start = output_start_list[0]
                outputs_end = output_end_list[0]
                for i in range(1, len(output_start_list)):
                    outputs_start = outputs_start + output_start_list[i]
                    outputs_end = outputs_end + output_end_list[i]
                outputs_start = outputs_start / folds_nums[idx]
                outputs_end = outputs_end / folds_nums[idx]
                outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
                outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
                ensemble_start.append(outputs_start)
                ensemble_end.append(outputs_end)
                ensemble_offset.append(offsets)
            
            for px, tweet in enumerate(orig_tweet):
                start_prob, end_prob = point_ensemble(ext(ensemble_start, px), ext(ensemble_end, px),
                                                      ext(ensemble_offset, px))
                start_idx = start_prob.argmax()
                end_idx = end_prob.argmax()
                if sentiment[px] == "neutral" or len(tweet.split()) < 2:
                    output_sentence = tweet
                else:
                    output_sentence = tweet[start_idx:end_idx]
                
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
    sample.to_csv(result_path + "submission.csv", index=False)
    
    sample.head()


print('CUDA_VISIBLE_DEVICES', os.environ["CUDA_VISIBLE_DEVICES"])
SEED = 1024
seed_everything(SEED)

ensemble1, test_data_loader1, folds_num1 = build_model([0, 1, 2, 3, 4], data_name='train', base_model='roberta',
                                                       name='roberta', head='linear', layers_used=2, num_layers=2,
                                                       batch_size=32)

ensemble2, test_data_loader2, folds_num2 = build_model([0, 1, 2, 3, 4], data_name='train', base_model='roberta',
                                                       name='roberta', head='linear', layers_used=2, num_layers=2,
                                                       batch_size=32)
result_path = f'results/ensemble_test_'
test_ensemble([ensemble1, ensemble2], [test_data_loader1, test_data_loader2], [folds_num1, folds_num1], result_path)
########
