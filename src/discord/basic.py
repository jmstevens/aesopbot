import discord
from discord.ext import commands
import random
from src.discord.giphy import Gif
import json
import sys
import os
import tensorflow as tf
import boto3
import json
import asyncio
import random
import src.models.aesop_gpt2 as gpt2
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import pickle
import math
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import json
import os
import time
from ftfy import fix_text
#:os.chdir('../')
import pickle
import numpy as np
import string, os
from gensim.models import KeyedVectors
import gensim.downloader as api
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from sklearn.model_selection import train_test_split
import random
import sys
from datetime import date
from collections import Counter
import matplotlib.pyplot as plt
from src.features.build import Lyrics
from src.features.transform_data import Transform
from random import shuffle
from tensorflow.python.framework import tensor_shape
from tokenizers import CharBPETokenizer, BertWordPieceTokenizer
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# from src.features.build import Provider
# model = inference()
description = '''Bot for aesop rock'''
bot = commands.Bot(command_prefix='?', description=description)
client = discord.Client()
# client.close()


@bot.event
async def on_ready():
    # global model
    # global data_reader
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')

@bot.command()
async def add(left : int, right : int):
    """Adds two numbers together."""
    await bot.say(left + right)

@bot.command()
async def roll(dice : str):
    """Rolls a dice in NdN format."""
    try:
        rolls, limit = map(int, dice.split('d'))
    except Exception:
        await bot.say('Format has to be in NdN!')
        return

    result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
    if len(result) < 2000:
        await bot.say(result)
    else:
        await bot.say('Result is too long. Look it up on wolfram alpha or something Kappa')

@bot.command(description='For when you wanna settle the score some other way')
async def choose(*choices : str):
    """Chooses between multiple choices."""
    await bot.say(random.choice(choices))

@bot.command()
async def repeat(times : int, content='repeating...'):
    """Repeats a message multiple times."""
    for i in range(times):
        await bot.say(content)

@bot.command()
async def joined(member : discord.Member):
    """Says when a member joined."""
    await bot.say('{0.name} joined in {0.joined_at}'.format(member))

@bot.group(pass_context=True)
async def cool(ctx):
    """Says if a user is cool.
    In reality this just checks if a subcommand is being invoked.
    """
    if ctx.invoked_subcommand is None:
        await bot.say('No, {0.subcommand_passed} is not cool'.format(ctx))

@cool.command(name='Pepo-Bot')
async def _bot():
    """Is the bot cool?"""
    await bot.say('Yes, the bot is cool.')

@bot.command()
async def subtract(left : int, right : int):
    """Adds two numbers together."""
    await bot.say(left - right)

@bot.command()
async def randomboi():
    """Posts a random dat boi"""
    boi = Gif('datboi', 50).random()
    await bot.say(boi)

@bot.command()
async def gif(*term: str):
    """Searches for and posts a gif from giphy"""
    term = " ".join(list(term))
    _gif = Gif(term, 50).random()
    await bot.say(_gif)

@bot.command(name='freestyle_topic')
async def freestyle_topic(ctx, context: str, seq_len: int, temperature: float, top_k: int,top_p: float):
    """Aesop Raps about a specific topic"""
    # global data_reader
    # print(term)
    term = "".join(list(context))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    special_tokens_dict = {'eos_token':'<END>','sep_token':'<NEWLINE>','bos_token':'<START>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    model = gpt2.Gpt2(6, 768, 8, 768, 128, vocab_size=tokenizer.vocab_size+3, tokenizer=tokenizer, optimizer="adam")#.load_model(filepath='checkpoint')
    model.create_optimizer()
    model.create_checkpoint_manager('checkpoint_longer')
    # model.create_summary_writer('logs')
    # model = model.load_model(filepath='checkpoint')
    # global data_reader
    # print(random.random(data_reader.lyrics))
    sample = model.sample_sequence(seq_len, context=context,temperature=temperature,top_k=top_k,top_p=top_p,nucleus_sampling=True)
    #sample = model.generate(data_reader, priming_text="hip hop ", sample=True, num_out=2000, temperature=.4)
    del model

    await ctx.send(sample)


@bot.command(name='freestyle_random')
async def freestyle_random(ctx, seq_len: int, temperature: float, top_k: int,top_p: float):
    """Aesop Raps: Random Aesop lyrics are fed in to seed the generator"""
    data_dir = 'data/processed/verses.txt'
    with open(data_dir, "rb") as fp:   # Unpickling
        lyrics = pickle.load(fp)
    arr = [' <NEWLINE> '.join([j for j in i.split(' \n ') if len(j) > 1 and '\n\n' != j]) for i in list(np.array(lyrics)) if len(i.split(' \n ')) > 0]#tokenizer = BertWordPieceTokenizer()
    #tokenizer.train(['data/processed/verses_encoded.txt'])
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    special_tokens_dict = {'eos_token':'<END>','sep_token':'<NEWLINE>','bos_token':'<START>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.encode(' <NEWLINE> '))
    tokenizer.save_pretrained('src/data/tokenizers')
    dataset = list()
    for verse in arr:
        tmp = list()
        verse = ' <START> ' + verse + ' <END> '
        verse_split = verse.split(' <NEWLINE> ')
        for line in verse_split:
            tmp = tmp + tokenizer.encode(line + ' <NEWLINE>', add_prefix_space=True)
        if tmp:
            dataset.append(tmp)
    context = random.choice(dataset)[:25]
    model = gpt2.Gpt2(12, 768, 16, 768, 16, vocab_size=tokenizer.vocab_size+3, tokenizer=tokenizer, optimizer="adam")
    model.create_optimizer()
    model.create_checkpoint_manager('checkpoint_long')
    model.create_summary_writer('logs')
    # global data_reader
    # print(random.random(data_reader.lyrics))
    sample = model.sample_sequence(300, context=context,temperature=temperature,top_k=top_k,top_p=top_p,nucleus_sampling=True)
    #sample = model.generate(data_reader, priming_text="hip hop ", sample=True, num_out=2000, temperature=.4)
    del model
    await ctx.send(sample)

# @bot.command()
# async def aesop_stats(*term: str):
#     global model
#     # global data_reader
#     stats = dict()
#     stats['model_info'] = model.__dict__
#     stats['iterator_info'] = model.__dict__
#     print(term)
#     if not isinstance(term, tuple):
#         await bot.say("I'm just a dumb bot right now.\nA real normie.\nA real yitbosy fucker.\n.\nsBut....\n...\n....\nI can only hand a single argument.\n\n\n...\n...BRO!")
#     else:
#         if len(term) > 1 and len(term) < 3:
#             if term[0] == 'listparams':
#                 print(list(stats.keys()))
#                 if term[1] == 'model':
#                     await bot.say(stats['model_info'])
#                 elif term[1] == 'iterator':
#                     await bot.say(stats['iterator_info'])
#                 else:
#                     print(list(stats.keys()))
#                     await bot.say("Pick from either {0} or {1}".format(stats.keys()))
#             elif term[0] == "list":
#                 try:
#                     config = stats[term[1]]
#                     await bot.say(config)
#                 except KeyError:
#                     await bot.say("Hey yo thats not a parameter my brain is not familiar with\nYou should try ?listparams")
#         else:
#             await bot.say("Idk man. You're on your own.\nYou do know I have documentaion right?.\nTry ?help")
#




# @bot.command()
# async def speak():
#     try:
#         await bot.say("You should hear some noise.")
#         os.system('afplay sound.mp3')  # Works only on Mac OS, sorry
#         os.remove('sound.mp3')
#     except FileNotFoundError:
#         await bot.say("No audiofile found! Please use the freestyle command to generate text")

with open('configs/config.json','r') as cfgFile:
    cfg = json.load(cfgFile)
    # discord_params = cfg['discord']['token']
# global model
# global data_reader
# tf.reset_default_graph()
# # term = " ".join(list(term))
# data_reader = Provider(cfg["model_params"]["LSTM"]["BATCH_SIZE"],
#                        cfg["model_params"]["LSTM"]["SEQUENCE_LENGTH"])
#
# vocabulary = data_reader.vocabulary
# sess = tf.Session()
# model = RNNModel(sess,
#                  vocabulary=vocabulary,
#                  batch_size=cfg["model_params"]["LSTM"]["BATCH_SIZE"],
#                  sequence_length=cfg["model_params"]["LSTM"]["SEQUENCE_LENGTH"],
#                  hidden_layer_size=cfg["model_params"]["LSTM"]["HIDDEN_LAYER_SIZE"],
#                  cells_size=cfg["model_params"]["LSTM"]["CELLS_SIZE"],
#                  keep_prob=cfg["model_params"]["LSTM"]["TRAIN_KEEP_PROB"],
#                  gradient_clip=cfg["model_params"]["LSTM"]["GRADIENT_CLIP"],
#                  starter_learning_rate=cfg["model_params"]["LSTM"]["STARTER_LEARNING_RATE"],
#                  decay_rate=cfg["model_params"]["LSTM"]["DECAY_RATE"],
#                  training=False
#                  )
#
# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# _num = str(max([int(i.replace('ckpt-','')) for i in list(set([i.split('.')[1] for i in os.listdir("src/data") if 'aesop.ckpt-' in i]))]))
# saver.restore(sess, "src/data/aesop.ckpt-{}".format(_num))
print('------')
bot.run(cfg['discord']['token'])
