#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import json
import torch
from chatbot import NeuralNet
from chatbot import bag_of_words, tokenize
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# In[ ]:


with open('intents.json', 'r') as f:
	intents = json.load(f)
with open('joke.json', 'r') as f:
	joke = json.load(f)
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name = "cerno"
print("Let's chat! type 'q' to exit")
while True:
	#
	setence = input("You: ")
	if setence == "q":
		break
	tokenized = tokenize(setence)
	X = bag_of_words(tokenized, all_words)
	X = X.reshape(1, X.shape[0])
	X = torch.from_numpy(X).to(device)
	output = model(X)
	_, predicted = torch.max(output, dim=1)
	tag = tags[predicted.item()]
	probs = torch.softmax(output, dim=1)
	prob = probs[0][predicted.item()]
	if prob.item() > 0.75:
		for intent in intents["intents"]:
			if tag == intent["tag"]:
				if(tag == "joke"):
					print(f"{bot_name}: {random.choice(joke['responses'])}")
				else:
					print(f"{bot_name}: {random.choice(intent['responses'])}")
	else:
		print(f"{bot_name}: I do not understand...")

