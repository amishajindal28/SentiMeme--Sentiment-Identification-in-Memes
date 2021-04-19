import numpy as np
import pandas as pd
from langdetect import detect
import re

#Import data
data = pd.read_pickle("./data/labels_pd_pickle")
data = data.drop(columns = ['text_ocr', 'humour', 'sarcasm', 'offensive', 'motivational'])

data["sentiment"] = data['overall_sentiment'].replace(
                        ['very_positive', 'positive', 'neutral', 'negative', 'very_negative'],
                        [1,1,0,-1,-1])

data["text_corrected"] = data["text_corrected"].str.lower()

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
data = data.dropna(subset=["text_corrected"])
data.to_pickle("./data/labels_pd_pickle2")