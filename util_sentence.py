import json
import random
from collections import OrderedDict
from tqdm import tqdm

import spacy
spacy.prefer_gpu()

nlp = spacy.load("en_core_web_sm")

import torch

def createSentenceCorpus(directory, shuffle = 1, seed = random.randint(1, 1000)):
	with open(directory, 'r') as f:
		dataset = f.readlines()
		
	sentenceDataset = []

	if shuffle:
		random.seed(seed)
		random.shuffle(dataset)

	num = len(dataset)

	for i, datum in enumerate(tqdm(dataset)):

		d = json.loads(datum)
		
		doc = nlp(d['text'])
		
		for sent in doc.sents:
			
			s = {'text':sent.text}

			sentenceDataset.append(json.dumps(s) + '\n')

	with open(directory, 'w') as f:
		f.writelines(sentenceDataset)
		

def sentencizeText(directory, shuffle = 1, seed = random.randint(1, 1000)):
	with open(directory, 'r') as f:
		dataset = f.readlines()
		
	if shuffle:
		random.seed(seed)
		random.shuffle(dataset)

	num = len(dataset)

	for i, datum in enumerate(tqdm(dataset)):

		d = json.loads(datum)
		
		doc = nlp(d['text'])
		
		d['sentences'] = [sent.text for sent in doc.sents]

		dataset[i] = json.dumps(d) + '\n'

	with open(directory, 'w') as f:
		f.writelines(dataset)

	return directory


def createSentenceVectorDataset(directory, sent2vec, shuffle = 1, seed = random.randint(1, 1000)):
	with open(directory, 'r') as f:
		dataset = f.readlines()
		
	if shuffle:
		random.seed(seed)
		random.shuffle(dataset)

	num = len(dataset)
	
	vectorDataset = []

	for _, datum in enumerate(tqdm(dataset)):

		d = json.loads(datum)
		
		v = dict((k, d[k]) for k in ['label'])
		
		v['sentences'] = [{sent:sent2vec(sent)} for sent in d['sentences']]
		
		vectorDataset.append(v)

	torch.save(vectorDataset, directory.replace('.json', '.pt'))

	return directory.replace('.json', '.pt')


class VectorPairDataset(torch.utils.data.Dataset):
	def __init__(self, directory, seed = random.randint(1, 1000), device = torch.device('cpu')):
		self.labels = labels
		self.list_IDs = list_IDs
		dataset = []
		labels = []
		rawDataset = torch.load(directory, map_location=device)
		
		num = len(rawDataset)
		
		for i, datum in enumerate(tqdm(rawDataset)):
			
			d_label = datum['label']
			d_sents = datum['sentences']
			if len(d_sents) > 1:
				
				for j in range(len(d_sents)):
					for k in range(j, len(d_sents)):
						pair = torch.cat((list[d_sents[j].values()][0], list[d_sents[k].values()][0])), 0)
						pair_label = 0 if d_label == 'real' else 1
						dataset.append(pair)
						labels.append(pair_label)
			
						
						
		

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		ID = self.list_IDs[index]

		# Load data and get label
		X = torch.load('data/' + ID + '.pt')
		y = self.labels[ID]

		return X, y
