from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models,util
from torch.utils.data import DataLoader
import time
from torch import nn
import pandas as pd
import torch
import os
import csv


## This is a big helper class, it is not essential to understand
class Sentence_tensor_generator:

	def setup(self, model_load_path,save_csv_path, text, article_idx, output_dim):
		self.text = text
		self.article_idx = article_idx
		self.output_dim = output_dim
		self.tensor_list = []
		self.save_csv_path = save_csv_path
		self.model = self.load_model(model_load_path)


	def setup_only_for_tensors(self, model_load_path, text, output_dim):
		self.text = text
		self.output_dim = output_dim
		self.tensor_list = []
		self.model = self.load_model(model_load_path)


	def generate_sentence_tensor(self):
		for item in self.text:
			self.tensor_list.append(self.model.encode(item, convert_to_numpy=True))


	def save_to_csv(self):
		path_text = self.save_csv_path + "/setences_text_"+str(self.output_dim)+".csv"
		path_tensors = self.save_csv_path + "/sentence_tensors_"+str(self.output_dim)+".csv"
		path_article_idx = self.save_csv_path + "/article_index_"+str(self.output_dim)+".csv"

		text_lst = []
		for sent in self.text:
			text_lst.append(sent.text)
		df = pd.DataFrame(text_lst, columns=['sentence'])
		df.to_csv(path_text)

		with open(path_tensors, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(self.tensor_list)

		with open(path_article_idx, 'w') as f:
			writer = csv.writer(f)
			writer.writerow(self.article_idx)


	def load_model(self, model_path):
		if not os.path.exists(model_path):
			word_embedding_model = models.Transformer('bert-base-uncased', 
													   max_seq_length=512
													   )
			pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
			dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
									   out_features=self.output_dim, 
									   activation_function=nn.Tanh()
									   )
			model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
			print("---- model for sentence tensor generator is created ----")
		else:
			model = SentenceTransformer(model_path)
			print("---- model for sentence tensor generator is loaded ----")
		return model

	def get_tensors(self):
		return self.tensor_list

	def get_text(self):
		return self.text

	def get_article_index(self):
		return self.article_idx

