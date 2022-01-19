# -*- coding: utf-8 -*-

import nltk
import re
import numpy as np
import pandas as pd
import unidecode
import spacy

nltk.download('punkt') # one time execution
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en import English

class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', ';', ':')

class AnswerRetriever():
    
    stop_words = set(stopwords.words('english'))
    TAG_RE = re.compile(r'<[^>]+>')
            
    def preprocess(self, text):
        #convert to lower case
        text = text.lower()
        
        #remove HTML tags (used for data taken from websites directly)
        #text = TAG_RE.sub('', text)
        
        #remove accents (in case google translate glitches)
        text = unidecode.unidecode(text)
        
        #remove punctuation (easier for posTag)
        text = re.sub(r'[^\w\s]', '', text)
        
        #remove stopwords
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in self.stop_words]
        return (' '.join(str(w) for w in filtered_sentence))
  
    def cosine_similarity_calc(self, vec_1,vec_2):
        sim = np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))
        return sim     

    def process(self, articles):
        processed_article = []
        prefix = " "
        for subarticle in articles: 
            for subsubarticle in subarticle:  
                prefix = subarticle[0]           
                if not (prefix==subsubarticle):
                    together = prefix+subsubarticle
                    processed_article.append(together)
        #display(processed_article)
        return processed_article
    
    def answer_process(self, text):
        #remove specific punctuations from the answer like :
        text = re.sub(r'[^\w\s,.()+=-?/]', '', text)
        return text
                                                          
    def extractAnswer(self):
        #tokenizer = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
        subarticles = re.split('\n[0-9]+\.', self.keyArticle)
        subsubarticles = []
        for subarticle in subarticles:  
            subsubarticles.append(re.split('\n[a-z]\.', subarticle))
        processed_articles = self.process(subsubarticles)
        ar_df = pd.DataFrame(processed_articles, columns=['sentences'])
        ar_df['processed_sentences'] = ar_df['sentences']
        ar_df['processed_sentences'] =    ar_df['processed_sentences'].apply(self.preprocess)
        ar_df['question'] = self.preprocess(self.question)
        embeddings = spacy.load('en_core_web_lg')
        print('hi1')
        ar_df['processed_similarity'] = ar_df.apply(lambda x:    self.cosine_similarity_calc(embeddings(x['processed_sentences']).vector,    embeddings(x['question']).vector), axis=1)  
        print('hi2')
        ar_df['similarity'] = ar_df.apply(lambda x:    self.cosine_similarity_calc(embeddings(x['sentences']).vector,    embeddings(x['question']).vector), axis=1)  
        ar_df = ar_df.sort_values(['processed_similarity'], ascending=False)
        ar_df['sentences'] = ar_df['sentences'].apply(self.answer_process)
        return ar_df
    
    def __init__(self, question, keyArticle):
        self.question = question
        self.keyArticle = keyArticle

