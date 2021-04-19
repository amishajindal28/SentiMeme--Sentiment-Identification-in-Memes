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
data["sentece_embed"] = model.encode(data["text_corrected"])


