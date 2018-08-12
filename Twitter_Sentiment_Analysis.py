# Twitter Sentiment Analysis

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv',encoding='latin-1')
test = pd.read_csv('test.csv',encoding='latin-1')
train.drop(['ItemID'],axis=1,inplace=True)
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
corpus1 = []
for i in range(0, 99989):
    review = re.sub('[^a-zA-Z]', ' ', train['SentimentText'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
