#https://github.com/UKPLab/sentence-transformers
from  sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import tensorflow as tf

#Import data and pre trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
size_embeddings = 768

data = pd.read_pickle("./data/labels_pd_pickle2")
#TODO: Make this line run jeje
not_working = []
working = []
for id_, sentence in enumerate(data["text_corrected"]):
    try:
        embed = model.encode(sentence)
        working.append(embed)
    except:
        not_working.append((id_, sentence))

#Shuffle data
text_embed = text_embed[np.random.permutation(np.arange(len(text_embed)))]
train_data, test_data = text_embed[:int(0.8*len(text_embed))], text_embed[:int(0.8*len(text_embed))]



