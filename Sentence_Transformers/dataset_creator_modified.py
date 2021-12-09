import random
from multiprocessing.dummy import Pool as ThreadPool
import re
import spacy


## This class creates dataset need for training Sentence Transformers
## @input: preprocessed/ reformated text file
## @output:
##     1) a text list including all the sentences in the text file
##	   2) a paired set including similar sentence pairs
##     3) a unpaired set including unsimilar sentence pairs
##     4) a article list including article number of each sentence in the text list
class dataset_creator():
	
	def __init__(self,file_address):
		self.file_address = file_address
		self.text = []
		self.paired_set = list()
		self.unpaired_set = list()
		self.label = list()
		self.sent1 = list()
		self.sent2 = list()
		self.article_text = dict()
		self.article_title = list()
		self.article_idx = list()
		self.nlp = spacy.load("en_core_web_sm")


	def create_dataset(self):
		pool = ThreadPool(8)
		
		
		with open(self.file_address) as f:
			# split text by articles
			article_number = 1
			text = [line.strip() for line in f]
			for i in range(0,len(text)):
				if re.match(r"[Aa]rticle[.]*", text[i]):
					lst = []
					for j in range(i+1, len(text)):
						if not re.match(r"[Aa]rticle[.]*", text[j]):
							lst.append(text[j])
						else:
							i=j
							break
					self.article_text[article_number] = lst
					article_number += 1

			# sentence segmentation
			# filter out all sentences with length less than 2
			for key in self.article_text.keys():
				value = self.article_text[key]
				# value_text = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'," ".join(line for line in value))
				value_text = self.nlp(" ".join(value)).sents
				value_text = [item for item in value_text if len(item)>3]
				self.text = self.text + value_text
				key_idx = [key]*len(value_text)
				self.article_idx = self.article_idx + key_idx


		# create paired sentences set
		txt_len = len(self.text)
		sentence_index1 = self.text[:txt_len-1]
		sentence_index2 = self.text[1:txt_len]
		self.paired_set = list(map(lambda x,y: [x,y], sentence_index1, sentence_index2))

		# create unpaired sentences set
		# using sentences from articles with interval > #sentences/5
		interval = txt_len/5
		index_rand1 = list(range(txt_len))*2
		index_rand2 = list(range(txt_len))*2
		random.shuffle(index_rand1)
		random.shuffle(index_rand2)
		self.unpaired_set = list(map(lambda x,y: [self.text[x],self.text[y]] if abs(x-y)>=interval else None, index_rand1, index_rand2))
		self.unpaired_set = [i for i in self.unpaired_set if i!=None]
		# unique_list = []
		# for x in self.unpaired_set:
		# 	if x not in unique_list:
		# 		unique_list.append(x)


		## create a concatenated label set for both sets
		## label paired sentences as 0.8 and unpaired 0.2
		self.label = [0.8]*len(self.paired_set) + [0.2]*len(self.unpaired_set)


	def combine_sets(self):
		pool = ThreadPool(8)
		self.sent1 = list(pool.map(lambda x: x[0], self.paired_set)) + list(pool.map(lambda y: y[0], self.unpaired_set))
		self.sent2 = list(pool.map(lambda x: x[1], self.paired_set)) + list(pool.map(lambda y: y[1], self.unpaired_set))


	def get_sentnce_pairs(self):
		return self.sent1, self.sent1


	def get_label(self):
		return self.label

	def get_text_and_article_list(self):
		return self.article_idx, self.text


	def reset(self):
		self.paired_set = list()
		self.unpaired_set = list()
		self.sent1 = list()
		self.sent2 = list()
		self.label = list()


# creator = dataset_creator("/Users/wangyangwu/Documents/Sentence_transformers/03_2021_trans.txt")
# creator.create_dataset()
# creator.combine_sets()
# print(creator.text[0].text)



