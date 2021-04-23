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

import gc
class VectorPairDataset(torch.utils.data.Dataset):
	def __init__(self, directory, seed = random.randint(1, 1000), device = torch.device('cpu')):
				
		rawDataset = torch.load(directory, map_location=device)
		homologous = []
		homologousLabels = []
		rawSamples = []
		
		
		num = len(rawDataset)

		for i, datum in enumerate(tqdm(rawDataset)):

			d_label = datum['label']
			d_sents = datum['sentences']

			for j in range(len(d_sents)):
				for k in range(j, min(20, len(d_sents))):
					pair = torch.cat((list(d_sents[j].values())[0], list(d_sents[k].values())[0]), 0)
					pair_label = 0 if d_label == 'real' else 1
					homologous.append(pair)
					homologousLabels.append(pair_label)

				rawSamples.append((list(d_sents[j].values())[0], i, d_label))


		self.homologous = homologous
		self.homologousLabels = homologousLabels
		
		self.rawSamples = rawSamples
		self.sampleNum = len(self.rawSamples)
		self.homologousNum = len(self.homologousLabels)
		
		self.nonHomologousNum = 7
		
		del rawDataset
		gc.collect()
	
		

	def __len__(self):

		return self.homologousNum + self.sampleNum * self.nonHomologousNum

	def __getitem__(self, index):
		
		if index < self.homologousNum:
			pair = self.homologous[index]
			pair_label = self.homologousLabels[index]
			
		else:
			sent1 = self.rawSamples[(index - self.homologousNum) // self.nonHomologousNum]
			sent2 = self.rawSamples[random.randint(1, self.sampleNum - 1)]
			pair = torch.cat((sent1[0], sent2[0]), 0)
			pair_label = 0 if sent1[2] == 'real' and sent2[2] == 'real' else 1

		return pair, pair_label
