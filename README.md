# FakeNewsNet-torchtext-dataset-json
A sample dataset in the format compatible with Torchtext

## Acknowledgement
The original dataset and downloader are from this [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) repository. The paper describing the dataset is [here](https://arxiv.org/abs/1809.01286). Shaun Toh helped pull the dataset, such that we can clean and sort the data.

## Why this dataset?
There are also articles that just straight up don't exist online anymore, not even on the wayback machine (the website must send their data to the wayback machine). Additionally, to make it compatible with Torchtext, necessary modifications have been conducted.

## Usage
There are around seven 'fields' in this dataset, namely ```title```, ```text```, ```tweets```(aggregated), ```spread``` number count, distinct ```user``` number to spread, publishing ```date``` of the article in float number, and finally ```label``` of the article whether it is 'real' or 'fake'. You may opt to ignore the meta data, if you would like to run a text summarization task using the ```text``` and ```title```. We ignore the rest of information, e.g., the source url, as it would greatly defeat the purpose of 'natural language' inference. There are websites that always intend to produce fake news. To load the (sample) dataset using Torchtext, try this:

```python
import spacy
import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator, RawField, LabelField

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


TEXT = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)

TITLE = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TWEETS = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
			
SPREAD = LabelField(dtype = torch.float, use_vocab=False, preprocessing=float) # yes, you use LabelField
USER = LabelField(dtype = torch.float, use_vocab=False, preprocessing=float)
DATE = LabelField(dtype = torch.float, use_vocab=False, preprocessing=float)
LABEL = LabelField(dtype = torch.float)

fields = {'title': ('title', TITLE), 'text': ('text', TEXT), 'tweets':('tweets', TWEETS), 'spread':('spread', SPREAD), 'user':('user', USER), 'date':('date', DATE), 'label':('label', LABEL)}

train_data, test_data = data.TabularDataset.splits(
                            path = 'your-path',
                            train = 'fakenewsnet-train.json',
                            test = 'fakenewsnet-test.json',
                            format = 'json',
                            fields = fields
)
train_data, valid_data = train_data.split()

TEXT.build_vocab(train_data)

TITLE.build_vocab(train_data)

TWEETS.build_vocab(train_data)

BATCH_SIZE = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.text),
     device = device)
```

by the way, you may want to split the dataset into 'train' and 'test' like this
```python
import random
random.seed(42)

directory = 'your-path/fakenewsnet.json'
with open(directory, 'r') as f:
    dataset = f.readlines()
shuffle = True
if shuffle:
    random.shuffle(dataset)
ratio = 0.3 # ratio of test set
num = len(dataset)
train_dataset = dataset[0:round(num * (1 - ratio))]
test_dataset = dataset[round(num * (1 - ratio)):]

with open(directory.replace('.json', '-train.json'), 'w') as g1:
    g1.writelines(train_dataset)
with open(directory.replace('.json', '-test.json'), 'w') as g2:
    g2.writelines(test_dataset)
print(len(dataset) == len(test_dataset) + len(train_dataset))
```

Note that publishing ```date``` of each entry is in a float number. We count the number of days between the actual date and 'Jan 1st, 1970', i.e., the Unix time ```0```. Then we divide this number by ```1e5```, resulting in a float number between 0 to 1. This is to avoid potential overflow problems. The publishing date for each article is not always clear, so we apply ```0.0``` to those without a clear date.
