from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models,util
from torch.utils.data import DataLoader
import time
from torch import nn
import torch
import os
import sys
from data_creator_advanced import dataset_creator_advanced
from fine_tuner import fine_tuner

## This is the main process of fine tuning
## It automatically fine tune the model with all dimesions in the "output_dim" list on
## the files stored in "directory" (needs to be changed on different OS envrionment)
## For future reference, you can put a new text file you want to train in the "directory" folder
## and remove all dimensions that you don't from the "outout_dim" list
def main():
	directory = "/Users/wangyangwu/Documents/Sentence_transformers/text_to_be_trained/"
	dest_directory = "/Users/wangyangwu/Documents/Sentence_transformers/text_trained/"
	model_directory = "/Users/wangyangwu/Documents/Sentence_transformers/saved_models/"
	file_names = [directory+filename for filename in os.listdir(directory) if filename.endswith(".txt")]
	dest_file_names = [dest_directory+filename for filename in os.listdir(directory) if filename.endswith(".txt")]
	# output_dim = [32,64,96,128,200,256,300,384,450,512]
	output_dim = [16]
	model_name = "fine_tuned_model_"
	fineTuner = fine_tuner()
	start = time.time()
	for i in range(len(output_dim)):
		print(f" --- Fine tuning model with output dimention {output_dim[i]}... ---")
		dim = output_dim[i]
		model_save_path = model_directory+model_name+str(dim)+"/"
		fineTuner.setup_output(dim, model_save_path)
		creator = dataset_creator_advanced(model_save_path,dim)
		for file in file_names:
			creator.set_file(file)
			creator.create_dataset()
			sent1 = creator.first_sentences
			sent2 = creator.second_sentences
			labels = creator.labels
			fineTuner.setup_data(sent1, sent2, labels)
			fineTuner.create_dataset_train_test(0, True, 32)
			fineTuner.fine_tune(2, 0, 0, True)
			# shutil.move(file, dest_file_names[i])
			creator.reset()
		print(f"---- Model is fine tuned ----")
	print(time.time()-start)

main()