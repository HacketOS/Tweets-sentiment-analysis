import csv
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tweepy
import nltk
import gensim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import prepare


def load_tweets(hashtag, since, count=100, filename='tweets.csv'):
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    csvFile = open(filename, 'w')
    csvWriter = csv.writer(csvFile)
    for tweet in tweepy.Cursor(api.search, q=hashtag,
                               lang="en",
                               since=since).items(count):
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


def prepare_tweets(tweets):
    prep = prepare.Prepare_machine(data=tweets)
    prep.remove_pattern(to_replace="b'RT")
    prep.remove_pattern(to_replace='b"RT')
    prep.remove_pattern(to_replace="b'")
    prep.remove_pattern(to_regular=r"http\S+")
    prep.remove_pattern(to_regular=r"@\S+")
    prep.remove_punctuation()
    prep.remove_numbers()
    prep.tokenize(reg=r'\w+')
    prep.remove_stop_words()
    prep.lemmatizing()
    return prep.data


def w2v_encoding(words, model, num_features, wset):
        featureVec = np.zeros(num_features, dtype="float32")
        nwords = 0
        for word in words:
            if word in wset:
                nwords = nwords + 1
                featureVec = np.add(featureVec, model[word])
        featureVec = np.divide(featureVec, nwords)
        return featureVec


def _main():
    DATE = '2019-09-02'
    HASHTAG = '#GlobalWarming'
    COUNT = 1000

    load_tweets(filename='tweets.csv', since=DATE, hashtag=HASHTAG, count=COUNT)
    data = pd.read_csv('tweets.csv', names=['time', 'tweet'])
    data['tweet'] = prepare_tweets(data['tweet'])
    w2v = gensim.models.Word2Vec.load('model.w2v')
    index2word_set = set(w2v.wv.index2word)
    encoded_tweets = []
    for tweet in data['tweet']:
        encoded_tweets.append(w2v_encoding(words=tweet, model=w2v,
                                           num_features=200,
                                           wset=index2word_set))
    encoded_tweets = np.array(encoded_tweets).reshape([-1, 200])
    invalid_rows = np.unique(np.argwhere(np.isnan(encoded_tweets))[:,0])
    encoded_tweets = np.delete(encoded_tweets, invalid_rows, axis=0)
    model = pickle.load(open('model.sav', 'rb'))
    print(model.predict(encoded_tweets).mean())

if __name__ == '__main__':
    _main()
