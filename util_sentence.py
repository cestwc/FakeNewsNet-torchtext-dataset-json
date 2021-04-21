import json
import random
from collections import OrderedDict

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

	for i, datum in enumerate(dataset):

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

	for i, datum in enumerate(dataset):

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

	for _, datum in enumerate(dataset):

		d = json.loads(datum)
		
		v = dict((k, d[k]) for k in ('label'))
		
		v['sentences'] = [{sent:sent2vec(sent)} for sent in d['sentences']]
		
		vectorDataset.append(v)

	torch.save(vectorDataset, directory.replace('.json', '.pt'))

	return directory.replace('.json', '.pt')
