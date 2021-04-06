# FakeNewsNet-torchtext-dataset-json
A sample dataset in the format compatible with Torchtext

## Acknowledgement
The original dataset and downloader are from this [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) repository. The paper describing the dataset is [here](https://arxiv.org/abs/1809.01286). Shaun Toh helped pull the dataset, such that we can clean and sort the data.

## Why this dataset?
There are also articles that just straight up don't exist online anymore, not even on the wayback machine (the website must send their data to the wayback machine). Additionally, to make it compatible with Torchtext, necessary modifications have been conducted.

The sample dataset provided here is a balanced sample, i.e., it contains 1024 articles labelled as 'fake', and another 1024 articles labelled as 'real'.

## Some features
The data pulled with original [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) occupy a enormous space on disk. However, these data are just seemingly bulky. There are a lot of 'id' information and repetitive tweets, which eat a lot of space. By the time we pulled the articles, the total number of **unfailing** articles is 20392, inclusive of both 'gossipcop' (19607) and 'politifact' (785). After removing those are with fewer than 25 words, there are altogether 19679 articles / entries. 4617 / 19679 are fake, and the rest 15062 / 19679 are real. You can find this 104MB version [here](https://drive.google.com/file/d/1-9TNx-0uIeLMXEOgtmYk7TMk57H_KDKD/view?usp=sharing)

Due to the size limit, we randomly sampled (```seed(42)```) 1024 entries from each side, and release this version. We preprocess the texts by removing URLs, mentions, repetitive title-like phrases in each tweet, and replacing unicodes with closest ASCII characters. Of course, more cleaning could be applied, though we feel it is ok for learning models so far.

## Usage
There are around nine 'fields' in this dataset, namely ```title```, ```text```, ```tweets```(aggregated), ```spread``` number count, distinct ```user``` number to spread, publish ```date``` of the article in float number, ```summary``` and ```keywords``` (aggregated), and finally ```label``` of the article whether it is 'real' or 'fake'. You may opt to ignore the meta data, if you would like to run a text summarization task using the ```text``` and ```title```. We ignore the rest of information, e.g., the source url, as it would greatly defeat the purpose of 'natural language' inference. There are websites that always intend to produce fake news. To load the (sample) dataset using Torchtext, try this:

```python
import spacy
import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, LabelField, BucketIterator

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
            lower = True, 
            include_lengths = True)

TWEETS = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)
	    
KEYWORDS = Field(tokenize = tokenize_en,
            lower = True, 
            include_lengths = True)

SUMMARY = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)
			
SPREAD = LabelField(dtype = torch.float, use_vocab=False, preprocessing=float) # yes, you use LabelField
USER = LabelField(dtype = torch.float, use_vocab=False, preprocessing=float)
DATE = LabelField(dtype = torch.float, use_vocab=False, preprocessing=float)
LABEL = LabelField(dtype = torch.float)

fields = {'title': ('title', TITLE), 'text': ('text', TEXT), 'tweets':('tweets', TWEETS), 'spread':('spread', SPREAD), 'user':('user', USER), 'date':('date', DATE), 'keywords':('keywords', KEYWORDS), 'summary':('summary', SUMMARY), 'label':('label', LABEL)}

train_data, test_data = data.TabularDataset.splits(
                            path = 'your-path',
                            train = 'fakenewsnet_sm-train.json',
                            test = 'fakenewsnet_sm-test.json',
                            format = 'json',
                            fields = fields
)
train_data, valid_data = train_data.split()

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data.text, train_data.title, train_data.summary, train_data.keywords, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

TITLE.vocab = TEXT.vocab
KEYWORDS.vocab = TEXT.vocab
SUMMARY.vocab = TEXT.vocab

TWEETS.build_vocab(train_data.text, train_data.title, train_data.tweets,
                   max_size = MAX_VOCAB_SIZE, 
                   vectors = "glove.twitter.27B.100d", 
                   unk_init = torch.Tensor.normal_)
		   
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

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

directory = 'your-path/fakenewsnet_sm.json' # _sm stands for 'small' dataset
with open(directory, 'r') as f:
    dataset = f.readlines()
shuffle = True
if shuffle:
    random.shuffle(dataset)
    
ratio = 0.3 # ratio of test set
num = len(dataset)

with open(directory.replace('.json', '-train.json'), 'w') as g1:
    g1.writelines(dataset[0:round(num * (1 - ratio))])
with open(directory.replace('.json', '-test.json'), 'w') as g2:
    g2.writelines(dataset[round(num * (1 - ratio)):])
```

Note that publish ```date``` of each entry is in a float number. We use the Unix time, and divide this number by ```1e10```, resulting in a float number between 0 to 1. This is to avoid potential overflow problems. The publishing date for each article is not always clear, so we apply ```0.0``` to those without a clear date.

Many fields, like ```keywords``` or ```summary```, might not exist, either.

## Final words
The **Usage** section introduced above can just nicely work with this [Tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb), which helped me a lot (and still helping) when I began to learn topics on languages. However, the accuracy is an LSTM, if applied here, will not be better than that of a random guess. The manual above is only to show you how you could load this dataset. You may want to take a look at the transformer [version](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb) of the tutorial series, which could give you at least 70% accuracy on the test set.
