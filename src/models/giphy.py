import time
import giphy_client
from giphy_client.rest import ApiException
from pprint import pprint
import json
import random as r
# create an instance of the API class
api_instance = giphy_client.DefaultApi()
with open('src/models/config.json') as f:
    config = json.load(f)
api_key = config["giphy"]["apikey"] # str | Giphy API Key

class Gif(object):
	def __init__(self, _term, limit):
		self.term = _term
		self.api = giphy_client.DefaultApi()
		self.key = api_key
		self.limit = limit
	def search(self):
		_results = self.api.gifs_search_get(api_key, self.term, limit = self.limit).to_dict()['data'][0]['url']
		return _results

	def random(self):
		_results = self.api.gifs_search_get(api_key, self.term).to_dict()['data']
		_results = _results[r.randrange(0, len(_results))]['url']
		return _results
