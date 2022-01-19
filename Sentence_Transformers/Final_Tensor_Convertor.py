import random
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from scipy import stats
import re
import spacy
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import os
import import_ipynb
import matplotlib.pyplot as plt
from data_creator_advanced import dataset_creator_advanced
import pandas as pd
import csv


## This class converts text sentences to tensors using fine-tuned model
class Sentence_Convertor:

	## This method defines several parameters:
	## model_path: path of fine_tuned model
	## save_csv_path: path to save the generated tensors
	## output_dim: output dimension of the model you choose
	def setup(self, model_path, save_csv_path, output_dim):
		self.model_path = model_path
		self.save_csv_path = save_csv_path
		self.output_dim = output_dim
		self.text = list()
		self.article_text = dict()
		self.article_idx = list()
		self.tensor_list = list()
		self.model = None
		self.nlp = spacy.load("en_core_web_sm")

	## This method is conjugate function for the whole converting process
	def convert_to_tensors(self, file_address):
		self.load_model()
		self.separate_and_reformat_sentences(file_address)
		self.generate_sentence_tensor()
		self.save_to_csv()

	## This method loads the fine-tuned model
	def load_model(self):
		if not os.path.exists(self.model_path):
			word_embedding_model = models.Transformer('bert-base-uncased', 
													   max_seq_length=512
													   )
			pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
			dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
									   out_features=self.output_dim, 
									   activation_function=nn.Tanh()
									   )
			self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
			print("---- model for sentence tensor generator is created ----")
		else:
			self.model = SentenceTransformer(self.model_path)
			print("---- model for sentence tensor generator is loaded ----")

	## This method preprocesses input text file:
	## 1. separate text into sentences
	## 2. reformat sentences and remove sentences length less than 3
	def separate_and_reformat_sentences(self, file_address):
	    with open(file_address) as f:
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
	            value_text = self.nlp(" ".join(value)).sents
	            value_text = [item.text for item in value_text if len(item)>3]
	            self.text = self.text + value_text
	            key_idx = [key]*len(value_text)
	            self.article_idx = self.article_idx + key_idx


	## This method uses model to convert individual sentences to tensors
	def generate_sentence_tensor(self):
		for item in self.text:
			self.tensor_list.append(self.model.encode(item, convert_to_numpy=True))


	## This method save the results to the specified path as csv files
	## Path needs to be changed for your OS environment
	def save_to_csv(self):
		path_text = self.save_csv_path + "/setences_text_"+str(self.output_dim)+".csv"
		path_tensors = self.save_csv_path + "/sentence_tensors_"+str(self.output_dim)+".csv"
		path_article_idx = self.save_csv_path + "/article_index_"+str(self.output_dim)+".csv"
		text_lst = []
		for sent in self.text:
			text_lst.append(sent)
		df = pd.DataFrame(text_lst, columns=['sentence'])
		df.to_csv(path_text)

		with open(path_tensors, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(self.tensor_list)

		with open(path_article_idx, 'w') as f:
			writer = csv.writer(f)
			writer.writerow(self.article_idx)


## This is the whole process of conversion
## Note: output_dim has to be the same as the dimension of the chosen model
def main():
	sc = Sentence_Convertor()
	model_path = "/Users/wangyangwu/Documents/Sentence_transformers/saved_models/fine_tuned_model_512"
	save_csv_path = "/Users/wangyangwu/Documents/Sentence_transformers/Results_Data"
	file_address = "/Users/wangyangwu/Documents/Sentence_transformers/text_to_be_trained/03_2021_trans.txt"
	output_dim = 512
	sc.setup(model_path, save_csv_path,output_dim)
	sc.convert_to_tensors(file_address)

