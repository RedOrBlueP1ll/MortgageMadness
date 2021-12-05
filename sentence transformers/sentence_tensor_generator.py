from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models,util
from torch.utils.data import DataLoader
from dataset_creator_modified import dataset_creator
import time
from torch import nn
import pandas as pd
import torch
import os
import csv

class Sentence_tensor_generator:

	def setup(self, model_load_path,save_csv_path, text, article_idx, output_dim):
		self.text = text
		self.article_idx = article_idx
		self.model = SentenceTransformer(model_load_path)
		self.tensor_list = []
		self.output_dim = output_dim
		self.save_csv_path = save_csv_path

	def generate_sentence_tensor(self):
		for item in self.text:
			self.tensor_list.append(self.model.encode(item.text, convert_to_numpy=True))

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


creator = dataset_creator("/Users/wangyangwu/Documents/Sentence_transformers/03_2021_trans.txt")
creator.create_dataset()
text = creator.get_text_and_article_list()[1]
article_idx = creator.get_text_and_article_list()[0]

output_dim = [32,64,128,256,384,512]

model_paths = ['fine_tuned_model_32/','fine_tuned_model_64/','fine_tuned_model_128/','fine_tuned_model_256/','fine_tuned_model_384/','fine_tuned_model_512/']
directory = '/Users/wangyangwu/Documents/Sentence_transformers/text_to_be_trained/'

save_csv_path = "/Users/wangyangwu/Documents/Sentence_transformers/Sentence_tensors"
model = Sentence_tensor_generator()

for i in range(len(model_paths)):
	full_model_path = directory+model_paths[i]
	dim = output_dim[i]
	model.setup(full_model_path, save_csv_path, text, article_idx, dim)
	model.generate_sentence_tensor()
	model.save_to_csv()
