import numpy as np
import pandas as pd
from langdetect import detect
import re
from  sentence_transformers import SentenceTransformer

#Import data and make changes to it
data = pd.read_pickle("./data/labels_pd_pickle")
data = data.drop(columns = ['text_ocr', 'humour', 'sarcasm', 'offensive', 'motivational'])

data["sentiment"] = data['overall_sentiment'].replace(
                        ['very_positive', 'positive', 'neutral', 'negative', 'very_negative'],
                        [1,1,0,-1,-1])

data["text_corrected"] = data["text_corrected"].str.lower()
data = data.dropna(subset=["text_corrected"])
data.to_pickle("./data/labels_pd_pickle2")

#This code was used to check the language of the text
#There is not any problem with the data and it is written in english
'''
def lang_detect(text):
    dicto = {}
    for word in text.split(" "):
        word_clean = re.sub('[^A-Za-z]+','', word)
        if len(word_clean) > 0:
            try:
                lang = detect(word_clean) 
                if lang not in dicto.keys(): dicto[lang] = 1
                else: dicto[lang] += 1
            except:
                print(word_clean)
    return sorted(dicto.items(), key=lambda item: item[1], reverse=True)[0][0]

data["lang"] = [lang_detect(text) if type(text) == str else "-1" for text in data["text_corrected"]  ]
not_english = data[data["lang"] != "en"] # most of them are actually english
data = data.iloc[:,:4]
'''

#Create the sentence embedding to have an initial model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
size_embeddings = 768
text_embed = [model.encode(sentence) for sentence in data["text_corrected"]]
text_embed = np.array(text_embed)
labels = data["sentiment"].to_numpy()

np.save("./data/text_embed", text_embed)
np.save("./data/labels_num", labels)