
import random
from multiprocessing.dummy import Pool as ThreadPool


## This class creates training data by extracting two sets of sentence
## pairs from text and include a label set:
## 1. Paired sentences (two sentences are followed by each other)
## 2. Unpaired sentences (two sentences are seperated)
## 3. labels: 0->paired / 1->unpaired
class dataset_creator():
	
	def __init__(text):
		self.text = text
		self.paired_set = list()
		self.unpaired_set = list()
		self.label = list()
		self.sent1 = list()
		self.sent2 = list()


	def create_dataset():
		pool = ThreadPool(8)
		txt_len = len(self.text)
		
		# create paired sentences set
		t1 = self.text[:txt_len-1]
		t2 = self.text[1:txt_len]
		self.paired_set = pool.map(lambda x,y: list((x,y)), t1, t2)

		# create unpaired sentences set
		rand1 = list(range(txt_len))
		rand2 = list(range(txt_len))
		random.shuffle(rand1)
		random.shuffle(rand2)
		self.unpaired_set = pool.map(lambda x,y: list((self.text[x],self.text[y])) if abs(x-y)>2 else None, rand1, rand2)

		## create a concatenated label set for both sets
		## label paired sentences as 1 and unpaired 0
		self.label = [0]*len(self.paired_set) + [1]*len(self.unpaired_set)


	def combine_sets(set1, set2):
		pool = ThreadPool(8)
		self.sent1 = pool.map(lambda x: x[0], set1) + pool.map(lambda y: y[0], set2)
		self.sent2 = pool.map(lambda x: x[1], set1) + pool.map(lambda y: y[1], set2)


	def get_sents():
		return self.sent1, self.sent1


	def get_label():
		return self.label


	def reset():
		self.paired_set = list()
		self.unpaired_set = list()
		self.sent1 = list()
		self.sent2 = list()
		self.label = list()



		