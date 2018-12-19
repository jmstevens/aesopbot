from src.features.genius import song, artist, api
from src.models.config import loadConfig
import json

with open('src/models/config.json','r') as cfgFile:
    cfg = json.load(cfgFile)['genius']



Genius = api.Genius(cfg['client_access_token'])
aesop = Genius.search_artist('Flatbush Zombies')


flatbush = json.loads('data/raw/lyrics.json')
# flatbush.keys()

print(aesop.__dict__)
