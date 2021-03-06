from tools.master import run, test, seed_everything, get_loss_fn, build_data, Model, collect
from tools.master import test_folds, train_folds
from tools.master import BERTModel, ELECTRAModel, RoBERTaModel, Albert, Embedding, XLNet
from tools.master import BERTLoader, RoBERTaLoader, AlbertLoader, XLNetLoader, SP_XLNetLoader
from tools.master import LinearHead, CNNHead, TransformerHead, LSTMHead, GRUHead, MixHead, SpanHead, SpanCNNHead, \
    SpanMixHead, StackCNNHead
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import argparse
import os

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
             'albert': AlbertLoader, 'embedding': BERTLoader, 'xlnet': SP_XLNetLoader}

head_list = {'linear': LinearHead, 'cnn': CNNHead, 'transformer': TransformerHead, 'lstm': LSTMHead, 'gru': GRUHead,
             'mix': MixHead, 'span_linear': SpanHead, 'span_cnn': SpanCNNHead, 'span_mix': SpanMixHead,
             'scnn': StackCNNHead}
schedule_list = {'linear_warmup': get_linear_schedule_with_warmup, 'cosine_warmup': get_cosine_schedule_with_warmup}
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1024, type=int)
parser.add_argument("--data", default='data/train_folds.csv', type=str,
                    choices=['data/train_folds.csv', 'data/extended_folds.csv'])
parser.add_argument("--model", default='roberta', type=str, choices=list(model_list.keys()))
parser.add_argument("--pretrained", default='roberta', type=str, choices=list(config.keys()))
parser.add_argument("--head", default='linear', type=str, choices=list(head_list.keys()))
parser.add_argument("--loss", default='ce', type=str)
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--schedule", default='linear_warmup', type=str, choices=['linear_warmup', 'cosine_warmup'])
parser.add_argument("--train", default=True, type=bool)

args = parser.parse_args()

print('CUDA_VISIBLE_DEVICES', os.environ["CUDA_VISIBLE_DEVICES"])

#  0
SEED = 1024
seed_everything(SEED)

#  1
data_name = 'train'
DATA = f'data/{data_name}_folds.csv'

#  2
MODEL = 'xlnet'
name = 'xlnet-base-cased'
base_model = model_list[MODEL](name=name)
data = data_list[MODEL]

#  3
HEAD = 'linear'
d_model = config[name]
layers_used = 2
head = head_list[HEAD](d_model, layers_used, num_layers=2)

#  4
LOSS = None
# loss_fn = get_loss_fn(ce=0.4, dst=0., span=0.4, jcd=0.1, lvs=0.4, jel=0.1, ksl=0.1)
loss_fn = get_loss_fn(ce=0.1, dst=0., span=0., jcd=0., lvs=0., jel=0., ksl=0.)

#  5
SCHEDULE = 'linear_warmup'
schedule = schedule_list[SCHEDULE]

#######
train_batch_size = 20
epochs = 3
lr = 3e-5

save_path = f'saved/{data_name}_{MODEL}_{d_model}_{HEAD}_'
result_path = f'results/{data_name}_{MODEL}_{d_model}_{HEAD}_'
print(save_path)
model = Model(base_model, head)
print('param num =', collect(model))
FOLDS = [0, 1, 2, 3, 4]

train_folds(FOLDS, DATA, data, model, train_batch_size, epochs, loss_fn, save_path,
            lr=lr, scheduler_fn=schedule, name=name, pretrained=True)

########
test_data = build_data("data/test.csv", data, train_batch_size, train_batch_size, name)
# test(model, test_data, save_path, result_path)
test_folds(FOLDS, model, test_data, save_path, result_path)
########

# run(0, DATA, data, model, train_batch_size, epochs, loss_fn, save_path, lr=lr, scheduler_fn=schedule, name=name)
# run(1, DATA, data, model, train_batch_size, epochs, loss_fn, save_path, lr=lr, scheduler_fn=schedule, name=name)
# run(2, DATA, data, model, train_batch_size, epochs, loss_fn, save_path, lr=lr, scheduler_fn=schedule, name=name)
# run(3, DATA, data, model, train_batch_size, epochs, loss_fn, save_path, lr=lr, scheduler_fn=schedule, name=name)
# run(4, DATA, data, model, train_batch_size, epochs, loss_fn, save_path, lr=lr, scheduler_fn=schedule, name=name)
#
# ########
# val_batch_size = train_batch_size
# test_data = build_data("data/test.csv", data, train_batch_size, val_batch_size, name)
# test(model, test_data, save_path, result_path)

########
