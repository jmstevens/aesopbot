import re
import tweepy
import json
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
import numpy as np

# from PIL import Image


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
# Load config file
with open('configs/config.json','r') as cfgFile:
    cfg = json.load(cfgFile)

consumer_key = cfg['twitter']['apiKey']
consumer_secret = cfg['twitter']["secretKey"]
access_token = cfg['twitter']["accessToken"]
access_token_secret = cfg['twitter']["accessTokenSecret"]

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)
data = api.rate_limit_status()

print(data['resources']['statuses']['/statuses/home_timeline'])
print(data['resources']['users']['/users/lookup'])
USERNAME = "realDonaldTrump"


def create_wordcloud(number, USERNAME):
    data = api.user_timeline(USERNAME, count=number)
    print([i.text for i in data])
    datasetJSON = [i._json for i in data]
    # print(mask)
    # Filter
    tweets = [i.text for i in data]
    tweets_url_removed = [re.sub(r"http\S+", '', i) for i in tweets]
    tokenizer = RegexpTokenizer(r'\w+')
    tweets_joined = tokenizer.tokenize(' '.join(tweets_url_removed))
    tweets_tknzr = TweetTokenizer(strip_handles=True, reduce_len=False, preserve_case=False)

    tweets_processed = [i for i in tweets_tknzr.tokenize(' '.join(tweets_joined)) if i != 'president' and i != 'rt']

    print(tweets_processed)
    # Word cloud
    wordcloud = WordCloud(width=512, height=512, background_color='white').generate(' '.join(tweets_processed))
    # create coloring from image
    wordcloud.to_file(f'test1_{USERNAME}.png')


if __name__ == "__main__":
    create_wordcloud(5000, USERNAME)

