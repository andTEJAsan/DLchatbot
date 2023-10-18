#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = nltk.stem.PorterStemmer()


# In[3]:


nltk.download('punkt')


# In[4]:


stemmer = PorterStemmer()

def tokenize(sentence):
	# tokenizes the sentence
	return nltk.word_tokenize(sentence)
def stem(word):
	# converts to the root word
	return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
	stemmed_sentence = [stem(w) for w in tokenized_sentence]
	bag_of_words = np.zeros(len(words), dtype=np.float32)
	for idx, w in enumerate(words):
		if w in stemmed_sentence:
			bag_of_words[idx] = 1.0
	return bag_of_words


# In[5]:


import torch
import torch.nn as nn


# In[6]:


class NeuralNet(nn.Module):
	def __init__(self, input_size,hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, num_classes)
		self.relu = nn.ReLU()
	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)
		# no activation and no softmax at the end
		return out


# In[ ]:




