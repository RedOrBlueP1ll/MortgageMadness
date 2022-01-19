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
from sentence_tensor_generator import Sentence_tensor_generator


## This class creates dataset needed for fine tuning Sentence Transformers
class dataset_creator_advanced():

    ## This method initializes path for input model and output dimension
    def __init__(self,model_path,output_dim): 
        self.text = []
        self.labels = list()
        self.first_sentences = list()
        self.second_sentences = list()
        self.initial_sentence_pairs = list()
        self.final_sentence_pairs = list()
        self.article_text = dict()
        self.article_title = list()
        self.article_idx = list()
        self.nlp = spacy.load("en_core_web_sm")
        self.similarity_matrix = list()
        self.sentence_matrix = list()
        self.model_path = model_path
        self.figure_count = 0
        self.output_dim = output_dim
        self.mean = list()
        self.sd = list()

    ## This method sets file address for the to-be-trained text file
    def set_file(self, file_address):
        self.file_address = file_address

    ## This method creates training dataset from the input text file
    def create_dataset(self):
        pool = ThreadPool(8)
        self.separate_and_reformat_sentences()
        self.calculate_sentence_pair_similarity()
        self.save_similarity_plots()
        self.create_sentence_pairs_with_labels()
        self.extract_sentence_pairs_for_each_quantile(len(self.initial_sentence_pairs)//200, 50)
        self.separate_sentences_labels()
        print("---- Training data is created ----")
        

    ## This method calculates similarities for all possible combinations of sentence pairs
    def calculate_sentence_pair_similarity(self):
        # get all sentence tensors
        sentence_generator = Sentence_tensor_generator()
        sentence_generator.setup_only_for_tensors(self.model_path, self.text, self.output_dim)
        sentence_generator.generate_sentence_tensor()
        sentence_tensors = sentence_generator.tensor_list
        for sent_idx in range(len(self.text)):
            this_sentence = self.text[sent_idx]
            other_sentences = [self.text[i] for i in range(len(self.text)) if i>sent_idx]
            sim_list = [util.cos_sim(sentence_tensors[sent_idx], sentence_tensors[i]).tolist()[0][0] for i in range(len(self.text)) if i>sent_idx]
            self.similarity_matrix.append(sim_list)
            self.sentence_matrix.append([this_sentence, other_sentences])

        
    ## This method saves histgram plots of generated similarity score
    def save_similarity_plots(self):
        result = list(np.concatenate(self.similarity_matrix).flat)
        self.mean.append(np.mean(result))
        self.sd.append(np.std(result))
        plt.figure(figsize=[12,5])
        plt.title("Model with output dimension - "+str(self.output_dim))
        plt.xlabel("After "+str(self.figure_count)+" trainings")
        sns_plot = sns.histplot(data=result)
        fig = sns_plot.get_figure()
        save_path = "dataset_histgram_figures/model_"+str(self.output_dim)+"_"+str(self.figure_count)+".png"
        fig.savefig(save_path)
        self.figure_count = (self.figure_count+1)%9
        if self.figure_count == 0:
            x = np.arange(len(self.mean))
            #plot mean similarity score
            plt.figure(figsize=[12,5])
            plt.title("Mean for model with output dimension "+str(self.output_dim))
            plt.xlabel("Number of PDF trained")
            plt.plot(x, self.mean)
            save_path_2 = "dataset_histgram_figures/model_"+str(self.output_dim)+"_mean.png"
            plt.savefig(save_path_2)
            #plot similarity SD
            plt.figure(figsize=[12,5])
            plt.title("Standard deviation for model with output dimension "+str(self.output_dim))
            plt.xlabel("Number of PDF trained")
            plt.plot(x, self.sd)
            save_path_3 = "dataset_histgram_figures/model_"+str(self.output_dim)+"_standard_deviation.png"
            plt.savefig(save_path_3)
            #reset mean and sd
            self.mean = list()
            self.sd = list()

        
    ## This method preprocesses text file: 
    ## 1. sentences separation
    ## 2. sentence reformation
    ## 3. remove sentences with lenght less than 3
    def separate_and_reformat_sentences(self):
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
                value_text = self.nlp(" ".join(value)).sents
                value_text = [item.text for item in value_text if len(item)>3]
                self.text = self.text + value_text
                key_idx = [key]*len(value_text)
                self.article_idx = self.article_idx + key_idx
        
        
    ## This method creates sentence pairs with labels 
    def create_sentence_pairs_with_labels(self):
        for i in range(len(self.sentence_matrix)):
            lst = [[self.sentence_matrix[i][0], self.sentence_matrix[i][1][j], self.similarity_matrix[i][j]] for j in range(len(self.similarity_matrix[i]))]
            self.initial_sentence_pairs += lst
      
    
    ## This method extracts samples from each quantile of similarity distribution
    ## The purpose of this method is to minimize the shortage of extreme similarity values
    ## num_sample: total number of samples to extract
    ## num_quantiles: number of quantiles from which sample are extract
    def extract_sentence_pairs_for_each_quantile(self, num_sample, num_quantiles):
        self.initial_sentence_pairs.sort(key=self.take_third) 
        quantiles = self.compute_quantiles(num_quantiles)
        num_items_in_each_quantile = len(self.initial_sentence_pairs)//num_quantiles
        num_samples_to_extract_from_each_quantile = num_sample//num_quantiles
        start_index = 0
        end_index = num_items_in_each_quantile
        final_sentence_pairs = []
        for _ in range(num_quantiles):
            final_sentence_pairs += random.sample(self.initial_sentence_pairs[start_index:end_index],num_samples_to_extract_from_each_quantile)
            start_index = end_index
            end_index = end_index + num_items_in_each_quantile
        self.final_sentence_pairs = final_sentence_pairs
        
    ## This method computes quantiles of the generated dataset
    def compute_quantiles(self,num_quantiles):
        result = list(np.concatenate(self.similarity_matrix).flat)
        quantiles = np.linspace(1/num_quantiles,1,num_quantiles)
        return np.quantile(result, quantiles)
    
    
    ## This method separates sentences and labels for training 
    def separate_sentences_labels(self):
        pool = ThreadPool(8)
        self.first_sentences = list(pool.map(lambda x: x[0], self.final_sentence_pairs))
        self.second_sentences = list(pool.map(lambda x: x[1], self.final_sentence_pairs))
        self.labels = list(pool.map(lambda x: x[2], self.final_sentence_pairs))
   

    ## This is a helper function for sorting a list
    def take_third(self,item):
        return item[2]

    ## This is a helper function for getting sentence pairs    
    def get_sentence_pairs(self):
        return self.first_sentences, self.second_sentences

    ## This is a helper function for getting labels
    def get_labels(self):
        return self.labels

    ## This is a helper function for getting text along with their article numbers
    def get_text_and_article_list(self):
        return self.article_idx, self.text

    ## This is a helper function for resetting all parameters
    def reset(self):
        self.text = []
        self.first_sentences = list()
        self.second_sentences = list()
        self.labels = list()
        self.sentence_matrix = list()
        self.similarity_matrix = list()
        self.initial_sentence_pairs = list()
        self.final_sentence_pairs = list()
        self.article_text = dict()
        self.article_title = list()
        self.article_idx = list()





