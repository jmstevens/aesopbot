import re
import tweepy
import json
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, TweetTokenizer

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

    # Filter
    tweets = [i.text for i in data if not i.text.startswith('RT')]
    tweets_url_removed = [re.sub(r"http\S+", '', i) for i in tweets]
    tokenizer = RegexpTokenizer(r'\w+')
    tweets_joined = tokenizer.tokenize(' '.join(tweets_url_removed))
    tweets_tknzr = TweetTokenizer(strip_handles=True, reduce_len=False, preserve_case=False)
    tweets_processed = tweets_tknzr.tokenize(' '.join(tweets_joined))
    print(tweets_processed)
    # Word cloud
    wordcloud = WordCloud().generate(' '.join(tweets_processed))
    wordcloud.to_file('test.png')

if __name__ == "__main__":
    create_wordcloud(200, USERNAME)
