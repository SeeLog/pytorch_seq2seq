import unicodedata
import re
from Lang import Lang
from my_utils import *

from tqdm import tqdm
import pickle

pat1 = re.compile("([.!?])")
pat2 = re.compile(r"[^a-zA-Z.!?]+")

def unicode2Ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalizeString(s):
    s = unicode2Ascii(s)
    s = pat1.sub(r" \1", s)
    s = pat2.sub(r" ", s)

    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    with open("data/train.%s" % lang1, "r", encoding="utf-8") as f:
        lines1 = f.readlines()

    with open("data/train.%s" % lang2, "r", encoding="utf-8") as f:
        lines2 = f.readlines()

    print("Normaraize...")
    for i, line in enumerate(my_tqdm(lines1)):
        lines1[i] = normalizeString(line)

    if reverse:
        pairs = list(zip(lines2, lines1))
        source_lang = Lang(lang2)
        target_lang = Lang(lang1)
    else:
        pairs = list(zip(lines1, lines2))
        source_lang = Lang(lang1)
        target_lang = Lang(lang2)

    token_pairs = list(zip(source_lang.tokenList, target_lang.tokenList))

    return source_lang, target_lang, pairs, token_pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    source_lang, target_lang, pairs, token_pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    for pair in my_tqdm(pairs):
        source_lang.addSentence(pair[0])
        target_lang.addSentence(pair[1])

    print("Counted words:")
    print(source_lang.name, source_lang.n_words)
    print(target_lang.name, target_lang.n_words)

    with open("Lang_%s.pkl" % lang1, "wb") as f:
        pickle.dump(source_lang, f)
    with open("Lang_%s.pkl" % lang2, "wb") as f:
        pickle.dump(target_lang, f)
    with open("pairs.pkl", "wb") as f:
        pickle.dump(pairs, f)
    with open("token_pairs.pkl", "wb") as f:
        pickle.dump(token_pairs, f)

    return source_lang, target_lang, pairs, token_pairs


def loadPkls(lang1, lang2):
    with open("Lang_%s.pkl" % lang1, "wb") as f:
        source_lang = pickle.dump(f)
    with open("Lang_%s.pkl" % lang2, "wb") as f:
        target_lang = pickle.dump(f)
    with open("pairs.pkl", "wb") as f:
        pairs = pickle.dump(f)

    return source_lang, target_lang, pairs


