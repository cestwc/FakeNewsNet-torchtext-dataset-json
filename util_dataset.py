import random

def split(directory, shuffle = 1, seed = random.randint(1, 1000)):
	with open(directory, 'r') as f:
		dataset = f.readlines()

	if shuffle:
		random.seed(seed)
		random.shuffle(dataset)
		
	ratio = 0.3 # ratio of test set
	num = len(dataset)
	train_dataset = dataset[0:round(num * (1 - ratio))]
	test_dataset = dataset[round(num * (1 - ratio)):]

	fileName = directory.split('/')[-1]

	with open(drivePath + fileName.replace('.json', '-train.json'), 'w') as g1:
		g1.writelines(train_dataset)
	with open(drivePath + fileName.replace('.json', '-test.json'), 'w') as g2:
		g2.writelines(test_dataset)

	return len(dataset) == len(test_dataset) + len(train_dataset)

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
