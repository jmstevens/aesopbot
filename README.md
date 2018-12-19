AesopBot
==============================
Discord chatbot which has the ability to freestyle (sort of like) Aesop Rock. Training is done with LSTM cells using Aesop's lyrics.
Can be (and probably will be) used to make a generic freestyling chatbot eventually.
=============================

Dependencies
=============================
Python 3.6_1

Create Environment
===========================
Clone the repo and run
```
make create_environment
```

If you see an error you may need to modify your Bashrc file with the following
```
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
venvwrap="virtualenvwrapper.sh"
/usr/bin/which -s $venvwrap
if [ $? -eq 0 ]; then
    venvwrap=`/usr/bin/which $venvwrap`
    source $venvwrap
fi
```

Activate Your Environment
==========================
```
workon aesopbot
```

Install Dependencies
=========================
```
make requirements
```

Download And Transform Data
========================
To download only aesop lyrics
```
make transform_aesop
```

To download the top 10 MC's lyrics for a basic hip hop bot
```
make transform
```

To Train The Model
=======================
```
make train
```

To Launch Bot
======================
```
make launch_bot
```
To make the bot freestyle do the following in discord
```
?freestyle {the word you want him to start with} {the number of "bars" you want to generate}
```

Note: 1 bar constitutes 10 words

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


#TODO
- [ ] Make the hyperparameters and other parameters for the ANN/Chatbot into a single config file that is read
  - [ ] Consolidate Discord Bot code into its own folder and maybe make it its own class?
    - [ ] also clean up chatbot code in general
  - [ ] Refactor the feature processing module completely
    - [ ] Look into other datasets
      - [ ] Lyric API or some other cleaner dataset
      - [ ] Reason:
        - [ ] Current dataset from kaggle is really inconsistant in how it organizes lyrics data, i.e. tough time seperating verses from choruses, etc.
      - [ ] Potential resources
        - [ ] PyRap (API to Rap Genius)
          https://github.com/dlarsen5/PyRap?files=1
    - [ ] Data processing <- TOP PRIORTY
      - [ ] Use Spacy or another NLP library to tokenize the data
      - [ ] Collect only versus for songs
      - [ ] Tokenize both end of lines and end of verses
      - [ ] Completely refactor the identification of MC"s to use for training
        - [ ] Count only stem words, truncate data to remove plurals, etc.
        - [ ] Identify Rhyme schemes, rhyme density, use only the first 20,000 lyrics
      - [ ] Create batches based on versus change the way sequencing is done?
    - [ ] Resource list
      - [ ] Drake NLP
        https://towardsdatascience.com/drake-using-natural-language-processing-to-understand-his-lyrics-49e54ace3662
      - [ ] Counting syllables
        https://stackoverflow.com/questions/46759492/syllable-count-in-python
      - [ ] Assesing rhyme density
        https://genius.com/posts/63-Introducing-rapmetricstm-the-birth-of-statistical-analysis-of-rap-lyrics
      - [ ] Rhyme detection
        https://github.com/kevin-brown/rhyme-detect/tree/master/rhyme_detect
      - [ ] Shakespear
        https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
- [ ] Allow checkpoint files to sync with s3 bucket for storage
  - [ ] Archecture changes
    - [ ] Add gradient clipping
    - [ ] Add ability to toggle between character and word embedding
  - [ ] Add a neural network so that aesop can have conversations
    - [ ] Resource List
      - [ ] Generic chatbot
        https://github.com/tensorlayer/seq2seq-chatbot/blob/master/main.py
      - [ ] Good blog post on making chatbots
        https://adeshpande3.github.io/How-I-Used-Deep-Learning-to-Train-a-Chatbot-to-Talk-Like-Me

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# aesopbot
