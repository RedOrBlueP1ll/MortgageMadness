from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models,util
from torch.utils.data import DataLoader
from data_creator_advanced import dataset_creator_advanced
import time
from torch import nn
import torch
import os
import sys
import shutil


class fine_tuner:

	## This method defines several varaibles needed for fine tuning
	## sentence1: first setences of all sentence pairs
	## sentence2: second setences of all sentence pairs
	## labels: similarity score of a pair ranging from 0.0-1.0
	def setup_data(self,sentences1, sentences2, labels):
		self.sentences1 = sentences1
		self.sentences2 = sentences2
		self.labels = labels
		self.training_data = []
		self.test_data = []
		self.training_dataloader = None
		self.model = None


	## This method sets up fine tuning hyperparameters
	## output_dim: dimension of output tensors
	## model_save_path: path to save the fine tuned model
	def setup_output(self,output_dim, model_save_path):
		self.output_dim = output_dim
		self.model_save_path = model_save_path


	## This method creates dataloader for training set
	## test_size: size of test data
	## if_shuffle: if shuffle the training data
	## batch_size: batch size of creating dataloader
	def create_dataset_train_test(self, test_size, if_shuffle, batch_size):
		train_size = int(len(self.sentences1)*(1-test_size))
		self.training_data = list(map(lambda x, y, z: InputExample(texts=[x,y], label=float(z)), self.sentences1[0:train_size], self.sentences2[0:train_size], self.labels[0:train_size]))
		self.training_dataloader = DataLoader(self.training_data, shuffle=if_shuffle, batch_size=batch_size)


	## This method is the main fine tuning function
	## epochs: number of epochs for training 
	## warmup_steps: number of warmup steps (default: 0)
	## evaluation_steps: number of evaluation (default: 0)
	## show_progress_bar: if progress bar is shown during training (default: true)
	def fine_tune(self, epochs, warmup_steps, evaluation_steps, show_progress_bar):
		
		if not os.path.exists(self.model_save_path):
			word_embedding_model = models.Transformer('bert-base-uncased', 
													   max_seq_length=512
													   )
			pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
			dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
									   out_features=self.output_dim, 
									   activation_function=nn.Tanh()
									   )
			self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
		else:
			self.model = SentenceTransformer(self.model_save_path)
			print(" -- model loaded --")

		loss_function = losses.CosineSimilarityLoss(self.model)
		self.model.fit(train_objectives=[(self.training_dataloader, loss_function)], 
				  epochs=epochs, 
				  warmup_steps=warmup_steps, 
				  show_progress_bar=show_progress_bar,
				  output_path=self.model_save_path,
				  # save_best_model = True,
				  # evaluator=evaluator
				  )

