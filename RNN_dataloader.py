import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


# load csv models
raw_train = pd.read_csv("gdrive/MyDrive/6120project/dataset_750/train.csv" ,encoding="utf-8")
raw_test = pd.read_csv("gdrive/MyDrive/6120project/dataset_750/test.csv" ,encoding="utf-8")
raw_val = pd.read_csv("gdrive/MyDrive/6120project/dataset_750/val.csv" ,encoding="utf-8")

#fixing the src_txt strings, removing the extra square brackets and the inverted commas that exist in the dataset
train = raw_train[['src_txt','tgt_txt']]
test = raw_test[['src_txt','tgt_txt']]
val = raw_val[['src_txt','tgt_txt']]
train['src_txt'] = train['src_txt'].str[2:-2] 
test['src_txt'] = test['src_txt'].str[2:-2] 
val['src_txt'] = val['src_txt'].str[2:-2] 

#creating csvs for the training, validation and testing data
train.to_csv("nachiket_train.csv")
val.to_csv("nachiket_val.csv")
test.to_csv("nachiket_test.csv")

#using the spacy tokenizer
spacy_eng = spacy.load("en")

# Define the tokeinze function
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# Define fields
text = Field(sequential = True,
             use_vocab=True,
             tokenize=tokenize_eng,
             init_token = '<sos>',
             eos_token='<eos>')

summary = Field(sequential=True,
                use_vocab=True,
                tokenize=tokenize_eng,
                init_token = '<sos>',
                eos_token='<eos>')

fields = {'src_txt':('t',text),'tgt_txt':('s',summary)}


#Load the data as torchtext datasets using TabularDataset
train_data, test_data, val_data = TabularDataset.splits(path = "./",
                                                        train = "nachiket_train.csv",
                                                        test = "nachiket_test.csv",
                                                        validation = "nachiket_val.csv",
                                                        format = "csv",
                                                        fields = fields)

# Build the vocabulary using glove embeddings
text.build_vocab(train_data,
                 min_freq=3,
                 vectors = 'glove.6B.100d')

#setting the summary.vocab as the same as the text.vocab
summary.vocab = text.vocab


batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the iterator, these are the same like dataloaders 
train_iterator, test_iterator, val_iterator = BucketIterator.splits((train_data,test_data,val_data),
                                                                    batch_size = batch_size,
                                                                    sort_within_batch = True,
                                                                    sort_key = lambda x: len(x.t),
                                                                    device = device)
