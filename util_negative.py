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
		
	for i, datum in enumerate(dataset):
		
		positive = json.loads(datum)
		
		negative = json.loads(dataset[random.choice([k for k in range(num) if k != i])])
		
		for field in fields:
			positive[field + '_neg'] = negative[field]
			
		dataset[i] = json.dumps(positive) + '\n'
		
	with open(directory, 'w') as f:
		f.writelines(dataset)
		
	return directory
