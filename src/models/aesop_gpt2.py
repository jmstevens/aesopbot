import tensorflow
import tensorflow as tf
import datetime
import torch
import matplotlib.pyplot as plt
import os, string
from torch.utils.data import Dataset
from transformers.modeling_tf_utils import TFTokenClassificationLoss
from transformers import BertTokenizer, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TFAutoModel, BertForSequenceClassification
from transformers import TFBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW
import transformers
from copy import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.features.text_preprocessing import clean_text
from tqdm import tqdm
import pickle
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time

RANDOM_SEED = 73
BATCH_SIZE = 4
EPOCHS = 1
MAX_LEN = 128

def load_data():
    data_dir = 'data/processed/verses.txt'
    with open(data_dir, "rb") as fp:   # Unpickling
        lyrics = pickle.load(fp)

    lyrics_clean = [clean_text(x).replace('\n','').strip().replace('  ',' ') for x in tqdm(lyrics)]
    return lyrics_clean

lyrics = load_data()[:1000]
# print(lyrics)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens_dict = {
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'pad_token': '<PAD>'}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

class FreestyleDataset(Dataset):
    def __init__(self, data, tokenizer, gpt2_type='gpt2', max_length=MAX_LEN):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for i in data:
            encodings_dict = tokenizer('<BOS>' + i + '<EOS>',
                                     truncation=True,
                                     max_length=max_length,
                                     padding='max_length')

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

freestyle_stanza_dataset = FreestyleDataset(lyrics, tokenizer, max_length=MAX_LEN)
def train_val_split(split, dataset):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    return train_size, val_size

freestyle_stanza_train_size, freestyle_stanza_val_size = train_val_split(0.8, freestyle_stanza_dataset)
freestyle_stanza_train_dataset, freestyle_stanza_val_dataset = random_split(freestyle_stanza_dataset, [freestyle_stanza_train_size, freestyle_stanza_val_size])

torch.cuda.manual_seed_all(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

freestyle_stanza_train_dataloader = DataLoader(freestyle_stanza_train_dataset,
                              sampler=RandomSampler(freestyle_stanza_train_dataset),
                              batch_size=BATCH_SIZE)

freestyle_stanza_val_dataloader = DataLoader(freestyle_stanza_val_dataset,
                            sampler=SequentialSampler(freestyle_stanza_val_dataset),
                            batch_size=BATCH_SIZE)


# helper function for logging time
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

# hyperparameters
learning_rate = 1e-4
eps = 1e-8
warmup_steps = 50

# create text generation seed prompt
device = torch.device('cuda')

prompt = "<BOS>"
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)

configuration = GPT2Config(vocab_size=len(tokenizer), n_positions=MAX_LEN).from_pretrained('gpt2', output_hidden_states=True)

freestyle_stanza_model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
freestyle_stanza_model.resize_token_embeddings(len(tokenizer))

freestyle_stanza_model.cuda()
optimizer = AdamW(freestyle_stanza_model.parameters(), lr=learning_rate, eps=eps)

total_steps = len(freestyle_stanza_train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

start_time = time.time()
freestyle_stanza_model = freestyle_stanza_model.to(device)

for epoch_i in range(0, EPOCHS):

    print(f'Epoch {epoch_i + 1} of {EPOCHS}')

    t0 = time.time()
    total_train_loss = 0
    freestyle_stanza_model.train()

    for step, batch in tqdm(enumerate(freestyle_stanza_train_dataloader),total=freestyle_stanza_train_size / BATCH_SIZE):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        freestyle_stanza_model.zero_grad()

        outputs = freestyle_stanza_model(b_input_ids,
                                    labels=b_labels,
                                    attention_mask=b_masks,
                                    token_type_ids=None)

        loss = outputs[0]

        batch_loss = loss.item()
        total_train_loss += batch_loss

        if step % 500 == 0 and step > 0:
            print(f'Average Training Loss: {total_train_loss / step}. Step Number: {step} / {len(freestyle_stanza_train_dataloader)}')

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(freestyle_stanza_train_dataloader)
    training_time = format_time(time.time() - t0)

    print(f'Average Training Loss: {avg_train_loss}. Epoch Training Time: {training_time}')

    t0 = time.time()

    freestyle_stanza_model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in freestyle_stanza_val_dataloader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        with torch.no_grad():

            outputs  = freestyle_stanza_model(b_input_ids,
                                         attention_mask=b_masks,
                                         labels=b_labels)

            loss = outputs[0]

        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(freestyle_stanza_val_dataloader)


    print(f'Average Validation Loss: {avg_val_loss}')

print(f'Total Training Time: {format_time(time.time()-start_time)}')

torch.save(freestyle_stanza_model.state_dict(), 'models/' + 'freestyle_stanza_model.pth')

freestyle_stanza_model.eval()

sample_outputs = freestyle_stanza_model.generate(
                                generated,
                                do_sample=True,
                                top_k=50,
                                max_length=MAX_LEN,
                                top_p=0.95,
                                num_return_sequences=3
                                )

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
