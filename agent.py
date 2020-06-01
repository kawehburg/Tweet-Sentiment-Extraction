from tools.master import train, test, seed_everything, get_loss_fn, build_data, Model, collect
from tools.master import BERTModel, ELECTRAModel, RoBERTaModel, Albert, Embedding
from tools.master import BERTLoader, RoBERTaLoader, AlbertLoader
from tools.master import LinearHead, CNNHead, TransformerHead
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import argparse
import os

model_list = {'bert': BERTModel, 'electra': ELECTRAModel, 'roberta': RoBERTaModel,
              'albert': Albert, 'embedding': Embedding}
pretrained_list = {'bert': ['bert-base-uncased', 'bert-large-uncased'],
                   'electra': ['google/electra-base-discriminator', 'google/electra-large-discriminator'],
                   'roberta': ['roberta'],
                   'albert': ['albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2'],
                   'embedding': ['embedding']}
config = {'bert-base-uncased': 768, 'bert-large-uncased': 1024,
          'google/electra-base-discriminator': 768, 'google/electra-large-discriminator': 1024,
          'roberta': 768,
          'albert-base-v2': 768, 'albert-large-v2': 1024, 'albert-xlarge-v2': 2048, 'albert-xxlarge-v2': 4096,
          'embedding': 256
          }

data_list = {'bert': BERTLoader, 'electra': BERTLoader, 'roberta': RoBERTaLoader,
             'albert': AlbertLoader, 'embedding': BERTLoader}

head_list = {'linear': LinearHead, 'cnn': CNNHead, 'transformer': TransformerHead}
schedule_list = {'linear_warmup': get_linear_schedule_with_warmup, 'cosine_warmup': get_cosine_schedule_with_warmup}
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1024, type=int)
parser.add_argument("--data", default='train', type=str, choices=['train', 'extended'])
parser.add_argument("--model", default='roberta', type=str, choices=list(model_list.keys()))
parser.add_argument("--pretrained", default='roberta', type=str, choices=list(config.keys()))
parser.add_argument("--head", default='linear', type=str, choices=list(head_list.keys()))
parser.add_argument("--loss", default='ce', type=str)
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--schedule", default='linear_warmup', type=str, choices=['linear_warmup', 'cosine_warmup'])
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--train", default=True, type=bool)

args = parser.parse_args()

print('CUDA_VISIBLE_DEVICES', os.environ["CUDA_VISIBLE_DEVICES"])
print(args)
#  0
SEED = args.seed
seed_everything(SEED)

#  1
data_name = args.data
DATA = f'data/{data_name}_folds.csv'

#  2
MODEL = args.model
name = args.pretrained
base_model = model_list[MODEL](name=name)
data = data_list[MODEL]

#  3
HEAD = args.head
d_model = config[name]
layers_used = 1
head = head_list[HEAD](d_model, layers_used, num_layers=6)

#  4
LOSS = args.loss
loss_fn = get_loss_fn(ce=1., jcd=0.)

#  5
SCHEDULE = args.schedule
schedule = schedule_list[SCHEDULE]

#######
train_batch_size = args.batch_size
val_batch_size = 16
epochs = args.epochs
lr = args.lr

save_path = f'saved/{data_name}_{MODEL}_{d_model}_{HEAD}_'
result_path = f'results/{data_name}_{MODEL}_{d_model}_{HEAD}_'
print(save_path)
model = Model(base_model, head)
print('param num =', collect(model))
if args.train:
    train(DATA, model, data, lr, train_batch_size, val_batch_size, name, loss_fn, schedule, epochs, save_path)

########
test_data = build_data("data/test.csv", data, train_batch_size, val_batch_size, name, fold=None)
test(model, test_data, save_path, result_path)

########
