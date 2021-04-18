import random
from collections import OrderedDict

def split(directory, sets = ['train', 'test'], ratio = 0.3, shuffle = 1, seed = random.randint(1, 1000)):
	with open(directory, 'r') as f:
		dataset = f.readlines()

	if shuffle:
		random.seed(seed)
		random.shuffle(dataset)
		
	sets.reverse()
	
	num = len(dataset)
	
	datasets = OrderedDict()
	
	for x in sets[:-1]:
		datasets[x] = dataset[round(num * (1 - ratio)):]
		dataset = dataset[0:round(num * (1 - ratio))]
		assert num == len(dataset) + len(datasets[x])
		num = len(dataset)
	datasets[sets[-1]] = dataset
	
	for key, value in datasets.items():
		with open(directory.replace('.json', '-' + key + '.json'), 'w') as f:
			f.writelines(value)
			
	return tuple(datasets.keys())[::-1]

def concatenate(*directories, destiny = 'concatenated.json', shuffle = 1, seed = random.randint(1, 1000)):
	dataset = []
	for directory in directories:
		with open(directory, 'r') as f:
			dataset += f.readlines()

	shuffle = 1
	if shuffle:
		random.shuffle(dataset)

	with open(destiny, 'w') as g:
		g.writelines(dataset)

	return len(dataset)
