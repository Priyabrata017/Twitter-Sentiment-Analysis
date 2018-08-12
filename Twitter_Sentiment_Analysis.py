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
for i in range(0, 299989):
    review = re.sub('[^a-zA-Z]', ' ', test['SentimentText'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 200)
X_train = cv.fit_transform(corpus).toarray()
y_train = train.iloc[:, 0].values
X_test= cv.fit_transform(corpus1).toarray()
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

sentiment=pd.DataFrame({"ItemID":test["ItemID"],"Sentiment":y_pred})
print(sentiment.info())
sentiment.to_csv('twitter.csv',index=False)
print("Exported")
