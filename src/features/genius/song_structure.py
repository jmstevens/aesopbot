"""From dlarsen5's PyRap song structure script.
https://github.com/dlarsen5/PyRap/blob/master/Song_Structure.py
"""
import itertools
import os
import pickle
import re
import json

# from nltk.corpus import stopwords

artists = ['Kendrick Lamar','Drake','Chance The Rapper','J. Cole','Logic','Future','Chief Keef','Eminem','Kanye West','JAY-Z','Big Sean',
                      'Lil Uzi Vert','Tyler, The Creator','Earl Sweatshirt','2 Chainz','G-Eazy','ScHoolboy Q','Young Thug','Joey Bada$$', 'Wu Tang Clan',
                      'Flatbush Zombies','A$AP Rocky','A$AP Ferg','Dumbfoundead','Tory Lanez','Waka Flocka Flame','Nas','A Tribe Called Quest','Vic Mensa',
                      '$UICIDEBOY$','Denzel Curry','Maxo Kream','Isaiah Rashad','Mike Stud','Mac Miller','Yonas','Childish Gambino','Rich Chigga',
                      'Three 6 Mafia','Azizi Gibson','RiFF RAFF','Lil Dicky','Lil Wayne','Tyga','Gucci Mane','Rick Ross','Asher Roth','Travis Scott','Migos','Rihanna',
                      'Bryson Tiller','21 Savage','Rae Sremmurd','French Montana','Miley Cyrus','XXXTENTACION','Lil Pump','Ski Mask the Slump God','Xavier Wulf',
                      'SmokePurpp','A Boogie Wit Da Hoodie','Playboi Carti','Ugly God','Wiz Khalifa','Justin Bieber','Beyoncé','Nicki Minaj','Meek Mill', 'Aesop Rock']


def get_lyrics_file():
    with open('data/raw/artist_lyrics.json') as f:
        data = json.load(f)
    return data


def get_verses(data):
    verse_lines = list()
    for k in data['artists']:
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
    return verse_lines

def clean_verses(verses):
    verses_list = []
    bad_characters = ['"',"'",'_','(',')','$',',','.','?','!','—','%','=']
    for song in verses:
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
    return verses_list

def segment_to_verses(verse_list):
    verses = []
    for i in clean:
        verse = ''
        for j in i:
            if isinstance(j, list):
                verse = verse + ' ' + ' '.join(j)
            else:
                verse = verse + ' ' + j
            verses.append(verse)
    return verses









# with open('data/raw/artist_lyrics.json') as f:
#     data = json.load(f)
#
# verse_lines = list()
# for k in data['artists']:
#     for v in k['songs']:
#         song = v['lyrics']
#         lines = song.splitlines()
#         for l in range(len(lines)):
#             title = [x.lower() for x in lines[l].replace('[', '').replace(']', '').split()]
#             if '[' in lines[l] and 'verse' in title:
#                 section_lines = []
#                 count = l + 1
#                 done = False
#                 while count < len(lines) and not done:
#                     if '[' not in lines[count]:
#                         if lines[count] != '':
#                             section_lines.append(lines[count])
#                         count += 1
#                     else:
#                         done = True
#                 verse_lines.append(section_lines)
#
#
#
#
#
def get_verses(all_lyrics,artist):
    #finds total verses, hooks, bridges, choruses written by a specific artist

    one_song_verse_lines = []
    one_song_chorus_lines = []
    one_song_hook_lines = []
    one_song_bridge_lines = []

    total_verse_lines = []
    total_chorus_lines = []
    total_hook_lines = []
    total_bridge_lines = []
    total_lines = []
    Songs = {}

    with open('data/raw/artist_lyrics.json') as f:
        data = json.load(f)

    verse_lines = list()
    for k in data['artists']:
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




    all_lyrics = data['artists']
    for _artist in all_lyrics:
        for artist,songs in _artist.items():
            for _song in songs:
                print(_song)
                if isinstance(_song, dict):
                    song_title = _song['title']
                    song_lyrics = _song['lyrics']
                    art = _song['artist']
                    clean_title = song_title.replace('(','').replace('.','').split()
                    if art == artist and 'Ft' not in clean_title:
                        lines = song_lyrics.splitlines()
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
                                total_verse_lines.append(section_lines)
                                one_song_verse_lines.append(section_lines)
                                print(section_lines)

                            elif '[' in lines[l] and 'chorus' in title:
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
                                total_chorus_lines.append(section_lines)
                                one_song_chorus_lines.append(section_lines)

                            elif '[' in lines[l] and 'hook' in title:
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
                                total_hook_lines.append(section_lines)
                                one_song_hook_lines.append(section_lines)

                            elif '[' in lines[l] and 'bridge' in title:
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
                                total_bridge_lines.append(section_lines)
                                one_song_bridge_lines.append(section_lines)

                artist_first_name = artist.split()[0].lower()

                total_lines = []
                one_song_verse_lines = []
                one_song_chorus_lines = []
                one_song_hook_lines = []
                one_song_bridge_lines = []

                if 'Ft' in clean_title:
                    lines = song_lyrics.splitlines()
                    for l in range(len(lines)):
                        title = [x.lower() for x in lines[l].replace('[','').replace(']','').replace('-','').replace(':','').split()]
                        if '[' in lines[l] and 'verse' in title and artist_first_name in title:
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
                            total_verse_lines.append(section_lines)
                            one_song_verse_lines.append(section_lines)

                        elif '[' in lines[l] and 'chorus' in title and artist_first_name in title:
                            section_lines = []
                            count = l + 1
                            done = False
                            while count < len(lines) and not done:
                                if '[' not in lines[count]:
                                    if lines[count] != '':
                                        section_lines.append(lines[count])
                                    count+=1
                                else:
                                    done = True
                            total_chorus_lines.append(section_lines)
                            one_song_chorus_lines.append(section_lines)

                        elif '[' in lines[l] and 'hook' in title and artist_first_name in title:
                            section_lines = []
                            count = l + 1
                            done = False
                            while count < len(lines) and not done:
                                if '[' not in lines[count]:
                                    if lines[count] != '':
                                        section_lines.append(lines[count])
                                    count+=1
                                else:
                                    done = True
                            total_hook_lines.append(section_lines)
                            one_song_hook_lines.append(section_lines)

                if len(one_song_verse_lines) > 0:
                    total_lines.append(total_verse_lines)

                if len(one_song_chorus_lines) > 0:
                    total_lines.append(total_chorus_lines)

                if len(one_song_hook_lines) > 0:
                    total_lines.append(total_hook_lines)

                if len(one_song_bridge_lines) > 0:
                    total_lines.append(total_bridge_lines)


                if len(total_lines) > 0:
                    Songs[song_title] = list(itertools.chain.from_iterable(total_lines))
    #FIXME: Songs has all duplicates
    Lines = {'Verses':total_verse_lines,'Choruses':total_chorus_lines,'Hooks':total_hook_lines,'Bridges':total_bridge_lines}

    return Lines, Songs

def clean_name(song_title):
    return re.sub(r'[^\x00-\x7F]+', ' ', song_title)

def clean_song_titles(song_dict):
    keys = [x.replace(u'\xa0', u' ').replace(u'\u200b',u'') for x in song_dict.keys()]
    new_keys = []
    featured_artists = []
    for k in keys:
        if '(' in k:
            if 'ft' in k.lower():
                new_k = k.split('(')[0][:-1]
                featured_artists.append(k.split('(')[1][:-1])
                new_keys.append(new_k)
            else:
                new_keys.append(k)
        else:
            new_keys.append(k)

    return new_keys,featured_artists

def make_one_list(song_lyrics):

    sentence_list = []
    bad_characters = ['"',"'",'_','(',')','$',',','.','?','!','—']
    bad_words = ['it', 'the', 'you', 'they', 'she', 'he', 'this', 'my', 'to', 'me', 'in', 'like', 'yeah', "you're",
                 "that's", "really", "couldn't",
                 'youre','get','want','come','uh','put','got','one','im',
                 'ran','em','right','gon','need','take','dont','every',
                 'turn','back','lets','better','look','see','til',
                 'aint','tryna','oh','still','yo',"don't","i'm",'gotta',
                 'know','go','yuh']
    stopword = set(stopwords.words('english'))

    total_stop_words = bad_words + list(stopword)

    lines = song_lyrics.splitlines()

    for line in lines:
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
            if w not in total_stop_words:
                words.append(w)
            new_word = []

        if words != []:
            sentence_list.append(words)

    return sentence_list

def Get_Artist_Lyrics(artist):
    song_lyrics_path = 'Song Lyrics/'

    artist_dict = {}


    for root,dirs,files in os.walk(song_lyrics_path):
        for f in files:
            name = root + f
            artist_dict[f.replace('_',' ')] = name

    all_lyrics = {}
    for art,path in artist_dict.items():
        with open(path,'rb') as f:
            all_lyrics[art] = pickle.load(f)

    Lines, Songs = get_verses(all_lyrics,artist)

    return Lines, Songs, all_lyrics

def Get_Lyrics():
    song_lyrics_path = 'data/raw/'

    artist_dict = {}


    for root,dirs,files in os.walk(song_lyrics_path):
        for f in files:
            name = root + f
            artist_dict[f.replace('_',' ')] = name
    json.loads(song_lyrics_path)
    all_lyrics = []

    for art,path in artist_dict.items():
        with open(path,'rb') as f:
            lyrics = pickle.load(f)
        for title,song_lyrics in lyrics.items():
            lyrics = make_one_list(song_lyrics)
            all_lyrics+=lyrics

    return all_lyrics

def Get_All_Lyrics(all_lyrics):

    sentence_list = []
    bad_characters = ['"',"'",'_','(',')','$',',','.','?','!','—']

    for artist,songs in all_lyrics.items():
        for title,song in songs.items():
            new_word = []
            separate = song.split()
            words = []
            for word in separate:
                for character in word:
                    character = character.lower()
                    if character not in bad_characters:
                        new_word.append(character)

                words.append(''.join(new_word))
                new_word = []

            sentence_list.append(words)

    return sentence_list

def get_song_structures(artist):
    Lines, Songs, All_lyrics = Get_Artist_Lyrics(artist)

    song_structures = {}

    for section, lines in Lines.items():
        rhyming_sections = []
        for lines_ in lines:
            rhyming_words = [line.split()[-2:] for line in lines_ ]
            rhyming_sections.append(rhyming_words)
        song_structures[section] = rhyming_sections

    return song_structures

song_structures = get_song_structures('Kendrick Lamar')
