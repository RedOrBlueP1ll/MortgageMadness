from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models,util
from torch.utils.data import DataLoader
from dataset_creator_modified import dataset_creator
import time
from torch import nn
import torch
import os
import sys


class fine_tuner:

	## import training data
	## sentences1: similar sentence pairs
	## sentences2: unsimilar sentence pairs
	## labels: 0.8->similar / 0.2->unsimilar
	def setup_data(self,sentences1, sentences2, labels):
		self.sentences1 = sentences1
		self.sentences2 = sentences2
		self.labels = labels
		self.training_data = []
		self.test_data = []
		self.training_dataloader = None
		self.model = None

	## set up training parameters
	## output_dim: dimension of output tensors
	## model_save_path: path to save the trained model
	def setup_output(self,output_dim, model_save_path):
		self.output_dim = output_dim
		self.model_save_path = model_save_path


	## create dataloader for training set
	## test_size: size of test data
	## if_shuffle: if shuffle the training data
	## batch_size: batch size of creating dataloader
	def create_dataset_train_test(self, test_size, if_shuffle, batch_size):

		train_size = int(len(self.sentences1)*(1-test_size))
		self.training_data = list(map(lambda x, y, z: InputExample(texts=[x,y], label=float(z)), self.sentences1[0:train_size], self.sentences2[0:train_size], self.labels[0:train_size]))
		self.training_dataloader = DataLoader(self.training_data, shuffle=if_shuffle, batch_size=batch_size)


	## fine tune model
	## epochs: number of epochs for training 
	## warmup_steps: number of warmup steps 
	## evaluation_steps: number of evaluation
	## show_progress_bar: if progress bar is shown during training
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


## start fine tuning models with different output dimensions
directory = "/Users/wangyangwu/Documents/Sentence_transformers/text_to_be_trained/"
file_names = [directory+filename for filename in os.listdir(directory) if filename.endswith(".txt")]
output_dim = [32,64,128,256,384,512]
model_save_path = ['fine_tuned_model_32/','fine_tuned_model_64/','fine_tuned_model_128/','fine_tuned_model_256/','fine_tuned_model_384/','fine_tuned_model_512/']
fine_tuner = fine_tuner()
start = time.time()
for i in range(len(output_dim)):
	print(f" --- Training model without output dimention {output_dim[i]}... ---")
	dim = output_dim[i]
	path = directory+model_save_path[i]
	fine_tuner.setup_output(dim, path)
	for file in file_names:
		creator = dataset_creator(file)
		creator.create_dataset()
		creator.combine_sets()
		sent1 = creator.sent1
		sent2 = creator.sent2
		labels = creator.label
		fine_tuner.setup_data(sent1, sent2, labels)
		fine_tuner.create_dataset_train_test(0, True, 16)
		fine_tuner.fine_tune(2, 0, 0, True)
	print(f" --- Model Trained --- ")
print(time.time()-start)
