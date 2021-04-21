import json
import random
from collections import OrderedDict

def addNegtiveSamples(directory, fields = ['text', 'label'], shuffle = 1, seed = random.randint(1, 1000)):
	with open(directory, 'r') as f:
		dataset = f.readlines()
		
	if shuffle:
		random.seed(seed)
		random.shuffle(dataset)
		
	num = len(dataset)
		
	for i, datum in enumerate(tqdm(dataset)):
		
		positive = json.loads(datum)
		
		negative = json.loads(dataset[random.choice([k for k in range(num) if k != i])])
		
		for field in fields:
			positive[field + '_neg'] = negative[field]
			
		dataset[i] = json.dumps(positive) + '\n'
		
	with open(directory, 'w') as f:
		f.writelines(dataset)
		
	return directory


import spacy
import numpy as np

def spacySimilarityTable(dataset, field = 'text'):
	
	nlp = spacy.load("en_core_web_md")  # make sure to use larger package!
	
	sentences = [json.loads(x)[field] for x in dataset]
	docs = [nlp(sent) for sent in sentences]
	num = len(dataset)
	table = np.zeros((num, num), dtype=float)
	for i in range(tqdm(num)):
		for j in range(i):
			similarity = docs[i].similarity(docs[j])
			table[i, j] = similarity
			table[j, i] = similarity
			
	ind = np.argsort(table, axis=0)[:, -10:]
	
	return ind
