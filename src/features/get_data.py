from src.features.genius.api import API, Genius
from configs import config
from src.features.genius.artist import Artist
from src.features.genius.song import Song
import json

class Build():
    """Wrapper for downloading and saving lyrics for a list of artists"""
    def __init__(self, artist_list):
        self.artist_list = artist_list
        with open('configs/config.json','r') as cfgFile:
            self.cfg = json.load(cfgFile)['genius']
        self.genius = Genius(self.cfg['client_access_token'])

    def build_artist(self, max_songs=None):
        if max_songs:
            self.artists = [self.genius.search_artist(i, max_songs=max_songs) for i in self.artist_list]
        else:
            self.artists = [self.genius.search_artist(i) for i in self.artist_list]

    def save(self):
        self.genius.save_artists(self.artists)

def main():
    with open('configs/config.json','r') as cfgFile:
        cfg = json.load(cfgFile)['artists']

    b = Build(cfg)
    b.build_artist()
    b.save()
