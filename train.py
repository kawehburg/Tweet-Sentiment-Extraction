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
from tools.master import collect
import utils

ROBERTA_PATH = "roberta"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json",
    merges_file=f"{ROBERTA_PATH}/merges.txt",
    lowercase=True,
    add_prefix_space=True
)
TRAINING_FILE = "data/extended_folds.csv"
MAX_LEN = 192
TRAIN_BATCH_SIZE = 20
VALID_BATCH_SIZE = 7
EPOCHS = 3
SAVE_HEAD = 'saved/extended_model_'
MODE = 'extended_'


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    """
    Processes the tweet and outputs the features necessary for model training and inference.

    Note: there are some differences between this and the BERT version (bert-case-uncased), mostly due to differences in token codes and special tokens
    """
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
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]
    
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


class TweetDataset:
    """
    Dataset which stores the tweets and returns them as processed features
    """
    
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self, item):
        data = process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )
        
        # Return the processed data where the lists are converted to `torch.tensor`s
        return {
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


class TweetModel(transformers.BertPreTrainedModel):
    """
    Model class that combines a pretrained bert model with a linear later
    """
    
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        # Load the pretrained BERT model
        self.roberta = transformers.RobertaModel.from_pretrained(ROBERTA_PATH, config=conf)
        # Set 10% dropout to be applied to the BERT backbone's output
        self.drop_out = nn.Dropout(0.1)
        # 768 is the dimensionality of bert-base-uncased's hidden representations
        # Multiplied by 2 since the forward pass concatenates the last two hidden representation layers
        # The output will have two dimensions ("start_logits", and "end_logits")
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        # Return the hidden states from the BERT backbone
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )  # bert_layers x bs x SL x (768 * 2)
        
        # Concatenate the last two hidden states
        # This is done since experiments have shown that just getting the last layer
        # gives out vectors that may be too taylored to the original BERT training objectives (MLM + NSP)
        # Sample explanation: https://bert-as-service.readthedocs.io/en/latest/section/faq.html#why-not-the-last-hidden-layer-why-second-to-last
        out = torch.cat((out[-1], out[-2]), dim=-1)  # bs x SL x (768 * 2)
        # Apply 10% dropout to the last 2 hidden states
        out = self.drop_out(out)  # bs x SL x (768 * 2)
        # The "dropped out" hidden vectors are now fed into the linear layer to output two scores
        logits = self.l0(out)  # bs x SL x 2
        
        # Splits the tensor into start_logits and end_logits
        # (bs x SL x 2) -> (bs x SL x 1), (bs x SL x 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)  # (bs x SL)
        end_logits = end_logits.squeeze(-1)  # (bs x SL)
        
        #####
        # start_logits = start_logits.masked_fill(mask == 0, -1e20)
        # end_logits = end_logits.masked_fill(mask == 0, -1e20)
        #####
        
        return start_logits, end_logits


def loss_fn(start_logits, end_logits, start_positions, end_positions, mask=None):
    c_loss = ce_loss(start_logits, end_logits, start_positions, end_positions)
    # use distance loss computed by jaccard score
    # d_loss = distance_loss(start_logits, end_logits, start_positions, end_positions, mask=mask)
    # loss = (c_loss + d_loss) / 2
    # loss = d_loss
    return c_loss


def ce_loss(start_logits, end_logits, start_positions, end_positions):
    """
        Return the sum of the cross entropy losses for both the start and end logits
        """
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def kl_loss(start_logits, end_logits, start_positions, end_positions, gama):
    loss_fct = nn.KLDivLoss()
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


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Trains the bert model on the twitter data
    """
    # Set model to training mode (dropout + sampled batch norm is activated)
    model.train()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    # Set tqdm to add loading screen and set the length
    tk0 = tqdm(data_loader, total=len(data_loader), ncols=80)
    
    # Train the model on each batch
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
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )  # (bs x SL), (bs x SL)
        # Calculate batch loss based on CrossEntropy
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, mask)
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
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,  # Full text of the px'th tweet in the batch
                target_string=selected_tweet,
                # Span containing the specified sentiment for the px'th tweet in the batch
                sentiment_val=tweet_sentiment,  # Sentiment of the px'th tweet in the batch
                idx_start=np.argmax(outputs_start[px, :]),  # Predicted start index for the px'th tweet in the batch
                idx_end=np.argmax(outputs_end[px, :]),  # Predicted end index for the px'th tweet in the batch
                offsets=offsets[px]  # Offsets for each of the tokens for the px'th tweet in the batch
            )
            # if tweet_sentiment == 'neutral':
            #     continue
            jaccard_scores.append(jaccard_score)
        # Update the jaccard score and loss
        # For details, refer to `AverageMeter` in https://www.kaggle.com/abhishek/utils
        avg = np.mean(jaccard_scores) if len(jaccard_scores) else 0
        jaccards.update(avg, len(jaccard_scores))
        losses.update(loss.item(), ids.size(0))
        # Print the average loss and jaccard score at the end of each batch
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


def calculate_jaccard_score(
        original_tweet,
        target_string,
        sentiment_val,
        idx_start,
        idx_end,
        offsets,
        verbose=False):
    """
    Calculate the jaccard score from the predicted span and the actual span for a batch of tweets
    """
    
    # A span's start index has to be greater than or equal to the end index
    # If this doesn't hold, the start index is set to equal the end index (the span is a single token)
    if idx_end < idx_start:
        idx_end = idx_start
    
    # Combine into a string the tokens that belong to the predicted span
    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        # If the token is not the last token in the tweet, and the ending offset of the current token is less
        # than the beginning offset of the following token, add a space.
        # Basically, add a space when the next token (word piece) corresponds to a new word
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "
    
    # Set the predicted output as the original tweet when the tweet's sentiment is "neutral", or the tweet only contains one word
    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet
    
    # Calculate the jaccard score between the predicted span, and the actual span
    # The IOU (intersection over union) approach is detailed in the utils module's `jaccard` function:
    # https://www.kaggle.com/abhishek/utils
    jac = utils.jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def eval_fn(data_loader, model, device):
    """
    Evaluation function to predict on the test set
    """
    # Set model to evaluation mode
    # I.e., turn off dropout and set batchnorm to use overall mean and variance (from training), rather than batch level mean and variance
    # Reference: https://github.com/pytorch/pytorch/issues/5406
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    # Turns off gradient calculations (https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch)
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), ncols=80)
        # Make predictions and calculate loss / jaccard score for each batch
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
            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            # Calculate loss for the batch
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, mask)
            # Apply softmax to the predicted logits for the start and end indexes
            # This converts the "logits" to "probability-like" scores
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            # Calculate jaccard scores for each tweet in the batch
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
            
            # Update running jaccard score and loss
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            # Print the running average loss and jaccard score
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg


def run(fold):
    """
    Train model for a speciied fold
    """
    # Read training csv
    dfx = pd.read_csv(TRAINING_FILE)
    
    # Set train validation set split
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    print('SIZE', len(df_train), len(df_valid))
    
    # Instantiate TweetDataset with training data
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )
    
    # Instantiate DataLoader with `train_dataset`
    # This is a generator that yields the dataset in batches
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=4
    )
    
    # Instantiate TweetDataset with validation data
    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )
    
    # Instantiate DataLoader with `valid_dataset`
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=2
    )
    
    # Set device as `cuda` (GPU)
    device = torch.device("cuda")
    # Load pretrained RoBERTa
    model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)
    # Output hidden states
    # This is important to set since we want to concatenate the hidden states from the last 2 BERT layers
    model_config.output_hidden_states = True
    # Instantiate our model with `model_config`
    model = TweetModel(conf=model_config)
    # Move the model to the GPU
    model.to(device)
    print(collect(model))
    # Calculate the number of training steps
    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    # Get the list of named parameters
    param_optimizer = list(model.named_parameters())
    # Specify parameters where weight decay shouldn't be applied
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Define two sets of parameters: those with weight decay, and those without
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    # Instantiate AdamW optimizer with our two sets of parameters, and a learning rate of 3e-5
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    # Create a scheduler to set the learning rate at each training step
    # "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period." (https://pytorch.org/docs/stable/optim.html)
    # Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    
    # Apply early stopping with patience of 2
    # This means to stop training new epochs when 2 rounds have passed without any improvement
    es = utils.EarlyStopping(patience=2, mode="max")
    # es = EarlyStopping(patience=2)
    print(f"Training is Starting for fold={fold}")
    
    # I'm training only for 3 epochs even though I specified 5!!!
    for epoch in range(EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=SAVE_HEAD + str(fold) + '.bin')
        if es.early_stop:
            print("Early stopping")
            break


fold = 0
# Read training csv
dfx = pd.read_csv(TRAINING_FILE)

# Set train validation set split
df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

# Instantiate TweetDataset with training data
train_dataset = TweetDataset(
    tweet=df_train.text.values,
    sentiment=df_train.sentiment.values,
    selected_text=df_train.selected_text.values
)

# Instantiate DataLoader with `train_dataset`
# This is a generator that yields the dataset in batches
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=4
)

# Instantiate TweetDataset with validation data
valid_dataset = TweetDataset(
    tweet=df_valid.text.values,
    sentiment=df_valid.sentiment.values,
    selected_text=df_valid.selected_text.values
)

# Instantiate DataLoader with `valid_dataset`
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=VALID_BATCH_SIZE,
    num_workers=2
)

run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)

df_test = pd.read_csv("data/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values

device = torch.device("cuda")
model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)
model_config.output_hidden_states = True

# Load each of the five trained models and move to GPU
model1 = TweetModel(conf=model_config)
model1.to(device)
model1.load_state_dict(torch.load(SAVE_HEAD + '0.bin'))
model1.eval()

model2 = TweetModel(conf=model_config)
model2.to(device)
model2.load_state_dict(torch.load(SAVE_HEAD + '1.bin'))
model2.eval()

model3 = TweetModel(conf=model_config)
model3.to(device)
model3.load_state_dict(torch.load(SAVE_HEAD + '2.bin'))
model3.eval()

model4 = TweetModel(conf=model_config)
model4.to(device)
model4.load_state_dict(torch.load(SAVE_HEAD + '3.bin'))
model4.eval()

model5 = TweetModel(conf=model_config)
model5.to(device)
model5.load_state_dict(torch.load(SAVE_HEAD + '4.bin'))
model5.eval()

final_output = []

# Instantiate TweetDataset with the test data
test_dataset = TweetDataset(
    tweet=df_test.text.values,
    sentiment=df_test.sentiment.values,
    selected_text=df_test.selected_text.values
)

# Instantiate DataLoader with `test_dataset`
data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)

# Turn of gradient calculations
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
        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start2, outputs_end2 = model2(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start3, outputs_end3 = model3(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start4, outputs_end4 = model4(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start5, outputs_end5 = model5(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
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
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            final_output.append(output_sentence)


# post-process trick:
# Note: This trick comes from: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942
# When the LB resets, this trick won't help
def post_process(selected):
    return " ".join(set(selected.lower().split()))


sample = pd.read_csv("data/sample_submission.csv")
sample.loc[:, 'selected_text'] = final_output
sample.selected_text = sample.selected_text.map(post_process)
sample.to_csv('results/' + MODE + "submission.csv", index=False)

sample.head()
