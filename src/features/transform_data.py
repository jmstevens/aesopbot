from src.features.genius.api import API, Genius
from configs import config
from src.features.genius.artist import Artist
from src.features.genius.song import Song
import json

with open('configs/config.json','r') as cfgFile:
    cfg = json.load(cfgFile)['genius']


Genius = api.Genius(cfg['client_access_token'])
aesop = Genius.search_artist('Flatbush Zombies')


flatbush = json.loads('data/raw/lyrics.json')
# flatbush.keys()

print(aesop.__dict__)

class Transform():
    def __init__(self):
        with open('data/raw/artist_lyrics.json') as f:
            self.data = json.load(f)
        self.get_verses()
        self.clean_verses()
        self.segment_to_verses()

    def get_verses(self):
        verse_lines = list()
        for k in self.data['artists']:
            for v in k['songs']:
                song = v['lyrics']
                lines = song.splitlines()
                for l in range(len(lines)):
                    title = [x.lower() for x in lines[l].replace('[', '').replace(']', '').split()]
                    if '[' in lines[l] and 'verse' in title:
                        section_lines = []
                        count = l + 1
                        done = False
                        while count < len(lines) and not done:
                            if '[' not in lines[count]:
                                if lines[count] != '':
                                    section_lines.append(lines[count])
                                count += 1
                            else:
                                done = True
                        verse_lines.append(section_lines)
        self.verse_lines = verse_lines
        return verse_lines

    def clean_verses(self):
        verses_list = []
        bad_characters = ['"',"'",'_','(',')','$',',','.','?','!','—']
        for song in self.verse_lines:
            verses = list()
            for line in song:
                if line == '\n':
                    continue
                if '[' in line:
                    continue
                new_word = []
                separate = line.split()
                words = []
                for word in separate:
                    for character in word:
                        character = character.lower()
                        if character not in bad_characters:
                            new_word.append(character)

                    w = ''.join(new_word)
                    # if w not in total_stop_words:
                    words.append(w)
                    new_word = []

                if words != []:
                    words = words + ['<eol>']
                    verses.append(words)
            verses = verses + ['<eov>']
            verses_list.append(verses)

        self.verses_list = verses_list
        return verses_list

    def segment_to_verses(self):
        verses = []
        for i in self.verses_list:
            verse = ''
            for j in i:
                if isinstance(j, list):
                    verse = verse + ' ' + ' '.join(j)
                else:
                    verse = verse + ' ' + j
                verses.append(verse.lstrip())
        self.verses = verses
        return verses
