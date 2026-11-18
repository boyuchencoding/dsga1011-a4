import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

import re

TYPO_PROB = 0.5  # probability that an eligible word receives a typo

# Neighbouring keys on a QWERTY keyboard
QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb",
    "h": "gyjn", "j": "hukm", "k": "jilm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Typo transformation: each alphabetic word with >= 4 letters is selected with
    # probability TYPO_PROB; in a selected word, one random letter is replaced by a
    # neighbouring key on the QWERTY keyboard (case preserved). Whitespace,
    # punctuation, HTML tags like <br /> and short words are left untouched.
    parts = re.split(r"(\s+|<[^>]*>)", example["text"])
    out = []
    for part in parts:
        if (
            part.isalpha()
            and part.isascii()
            and len(part) >= 4
            and random.random() < TYPO_PROB
        ):
            idx = random.randrange(len(part))
            ch = part[idx]
            neighbors = QWERTY_NEIGHBORS.get(ch.lower())
            if neighbors:
                new = random.choice(neighbors)
                new = new.upper() if ch.isupper() else new
                part = part[:idx] + new + part[idx + 1:]
        out.append(part)
    example["text"] = "".join(out)

    ##### YOUR CODE ENDS HERE ######

    return example
