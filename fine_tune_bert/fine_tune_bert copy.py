import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split,RandomSampler, SequentialSampler
import tensorflow as tf
from functools import reduce
from transformers import BertTokenizer, BertForPreTraining, BertForNextSentencePrediction, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
from datasets import load_metric
import os


class BERT_fine_tuner():
	def __init__(self, sent1, sent2, label):
		self.sent1 = sent1 #first sentences
		self.sent2 = sent2 #second sentences
		self.label = label #labels: 0->paired / 1->unpaired
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) #load BERT tokenizer
		self.trainDataLoader #Pytorch dataloader for training set
		self.testDataLoader #Pytorch dataloader for test set
		self.MAX_LEN #max length of tensor
		self.input_dict #tokenized datasets
		self.model #pretrained BERT model
		self.pool = ThreadPool(8) #thread pool to acelerate computation
		# BERT authors recommend the following hyperparameters:
		self.batch_size = [16,32]
		self.epochs = [2,3,4]
		self.learning_rate = [5e-5, 3e-5, 2e-5] 
		self.eps = [1e-8, 1e-6] #default epsilon is 1e-6
		self.train_loss = list() #array to store training losses of all epochs
		self.test_loss = list() #array to store test losses of all epochs
		self.metrics #variable to store accuracies of all batches
		self.model_dir = './fine_tuned_model/' #directory to save fine tuned model



	## Tokenize datasets using the same tokenizer as pretrained BERT model
	## Pad tensors to the max length of the tokenized sentence in the dataset
	## Truncation is applied
	def tokenize():
		self.input_dict = self.tokenizer(
										self.sent1,
										self.sent2,
										max_length = self.MAX_LEN,
										truncation = True,
										padding = 'max_length',
										return_tensors = 'pt'
										)
		self.input_tensors['label'] = torch.LongTensor([self.label]).T


	## Split the dataset into training set and test set by 8:2
	## Create pytorch dataloader for both datasets 
	def createDataLodaer():
		dataset = self.thisDataSet(self.input_dict)
		train, test = random_split(dataset,[0.8*len(dataset), 0.2*len(dataset)])
		self.trainDataLoader = DataLoader(train, 
									 batch_size=self.batch_size[1], 
									 sampler = RandomSampler(train))
		self.testDataLoader = DataLoader(test, 
									 batch_size=self.batch_size[1], 
									 sample = SequentialSampler(test))


	## Fine tune BERT model
	def fineTune():
		# if previous fine tuned model exists, load the model
		# else, load a pretrained model
		if not os.path.exists(self.model_dir):
			self.model = BertForNextSentencePrediction.from_pretrained(
							"bert-base-uncased",
							num_labels = 2)
		else:
			self.model = BertForNextSentencePrediction.from_pretrained(
							self.model_dir)
		# check if GPU is available and move model to the available device
		if torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
		self.model.to(device)
		# create adam optimizer AdamW with small eps
		optimizer = AdamW(model.parameters(),
						  eps=self.eps[1],
						  lr=self.learning_rate[0])
		# create a linear schedule for learning rate
		# with linear schedule, learning rate increases from 0 to specified lr
		# during warmup, and then decreases to 0 after the warmup
		# here we don't assign warmup therefore learning rate starts from 
		# specified lr
		training_steps = len(self.trainDataLoader)*self.epochs[0]
		schedule = get_linear_schedule_with_warmup(optimizer,
												   num_warmup_steps=0,
												   nnum_training_steps=training_steps)
		#============================ train section ============================#
		# start training
		self.model.train()  #put model in training mode (it doesn't start training)
		for epoch in self.epochs[0]:
			loop_train = tqdm(self.trainDataLoader, leave=True)
			for batch in loop_train:
				# obtain all tensors from this batch and move them to CPU/GPU
				input_ids_train = batch[0].to(device)
				token_type_ids_train = batch[1].to(device)
				attention_mask_train = batch[2].to(device)
				labels_train = batch[3].to(device)
				self.model.zero_grad() #set gradients to 0
				result_train = self.model(input_ids_train,
							   token_type_ids=token_type_ids_train,
							   attention_mask=attention_mask_train,
							   labels=labels_train)
				loss_train = result_train.loss
				self.train_loss.append(loss_train.item()) #record train loss
				loss_train.backward() #backpropagation
				optimizer.step() #update parameters
				schedule.step() #update learning rate
				#show process
				loop.set_description(f"Epoch {epoch}")
				loop.set_postfix(loss=loss_train.item())
		#============================= test section ============================#
		metric = load_metric("accuracy") 
		self.model.eval() #switch model to evaluation mode
		loop_train = tqdm(self.testDataLoader, leave=True)
		for batch in loop_train:
			# move tensors to CPU/GPU
			input_ids_test = batch[0].to(device)
			token_type_ids_test = batch[1].to(device)
			attention_mask_test = batch[2].to(device)
			labels_test = batch[3].to(device)
			self.model.zero_grad() #set gradients to 0
			#test without computing gradients
			with torch.no_grad():
				result_test = self.model(input_ids_test,
							   token_type_ids=token_type_ids_test,
							   attention_mask=attention_mask_test,
							   labels=labels_test)
			loss_test = result_test.loss
			logits_test = result_test.logits
			predictions = torch.argmax(logits, dim=-1)
			metrix.add_batch(predictions=predictions, references=batch['labels']) #add a batch pf predictions and references to the metrics stack
			self.test_loss.append(loss_test.item()) #record test loss
			#show process
			loop_train.set_description(f"Batch {batch}")
			loop_train.set_postfix(loss=loss_test.item())
		self.metrics = metric.compute() #get the metrics of evaluation
		#============================= save model ============================#
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		self.model.save_pretrained(self.model_dir)
		self.tokenizer.save_pretrained(self.model_dir)


	## Plot loss
	def plot_loss(self):
		plt.plot(self.train_loss, "r-", label="training loss")
		plt.plot(self.test_loss, "g-", label="test loss")
		plt.legend()
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.show()


	## Get the max length of all tokenized sentenses 
	## The purpose of this is to pad to the actual max length of the sentences 
	## in the dataset instead of using 512
	def getMaxLen(self):
		self.MAX_LEN = self.pool.reduce(lambda x,y: max(self.tokenizer.encode(x,add_special_tokens=True),
											  			self.tokenizer.encode(y, add_special_tokens=True)),
											  		self.sent1)
		self.MAX_LEN = self.pool.reduce(lambda x,y: max(self.tokenizer.encode(x,add_special_tokens=True),
											  			self.tokenizer.encode(y, add_special_tokens=True)),
											  		self.sent2)
		self.MAX_LEN = min(self.MAX_LEN, 512)


	## Inner class for Pytorch dataset
	class thisDataSet(Dataset):
		def __init__(self, inputs):
			self.inputs = inputs
		def __len__(self):
			return len(self.inputs[input_ids])
		def __getitem__(self, idx):
			return {key: inputs[key][idx] for key in self.inputs.keys()}


