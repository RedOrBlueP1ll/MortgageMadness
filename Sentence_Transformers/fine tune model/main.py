from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models,util
from torch.utils.data import DataLoader
import time
from torch import nn
import torch
import os
import sys
from dataset_creator_modified import dataset_creator
from fine_tuner import fine_tuner

## This is the main process of fine tuning
## It automatically fine tune the model with all dimesions in the "output_dim" list on
## the files stored in "directory" (needs to be changed on different OS envrionment)
## For future reference, you can put a new text file you want to train in the "directory" folder
## and remove all dimensions that you don't from the "outout_dim" list
def main():
	directory = "/Users/wangyangwu/Documents/Sentence_transformers/text_to_be_trained/"
	file_names = [directory+filename for filename in os.listdir(directory) if filename.endswith(".txt")]
	output_dim = [32,64,96,128,256,300,384,450,512] #
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