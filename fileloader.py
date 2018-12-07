import unicodedata
import re
from Lang import *
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

def readLangs(lang1, lang2, valid=False, reverse=False):
    print("Reading lines...")

    if not valid:
        with open("data/train.%s" % lang1, "r", encoding="utf-8") as f:
            lines1 = f.readlines()

        with open("data/train.%s" % lang2, "r", encoding="utf-8") as f:
            lines2 = f.readlines()
    else:
        with open("data/val.%s" % lang1, "r", encoding="utf-8") as f:
            lines1 = f.readlines()

        with open("data/val.%s" % lang2, "r", encoding="utf-8") as f:
            lines2 = f.readlines()

    print("Normaraize...")
    for i, line in enumerate(my_tqdm(lines1)):
        lines1[i] = normalizeString(line)

    for i, line in enumerate(my_tqdm(lines2)):
        lines2[i] = line.replace('\n', '')

    if reverse:
        pairs = list(zip(lines2, lines1))
        source_lang = Lang(lang2)
        target_lang = Lang(lang1)
    else:
        pairs = list(zip(lines1, lines2))
        source_lang = Lang(lang1)
        target_lang = Lang(lang2)

    return source_lang, target_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, valid=False, reverse=False):
    source_lang, target_lang, pairs = readLangs(lang1, lang2, valid, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    for pair in my_tqdm(pairs):
        source_lang.addSentence(pair[0])
        target_lang.addSentence(pair[1])

    token_pairs = list(zip(source_lang.tokenList, target_lang.tokenList))

    print("Counted words:")
    print(source_lang.name, source_lang.n_words)
    print(target_lang.name, target_lang.n_words)

    del source_lang.tokenList
    del target_lang.tokenList
    del source_lang.mecab
    del target_lang.mecab

    if not valid:
        with open("./pkls/train_Lang_%s.pkl" % lang1, "wb") as f:
            pickle.dump(source_lang, f)
        with open("./pkls/train_Lang_%s.pkl" % lang2, "wb") as f:
            pickle.dump(target_lang, f)
        with open("./pkls/train_pairs.pkl", "wb") as f:
            pickle.dump(pairs, f)
        with open("./pkls/train_token_pairs.pkl", "wb") as f:
            pickle.dump(token_pairs, f)
    else:
        with open("./pkls/eval_pairs.pkl", "wb") as f:
            pickle.dump(pairs, f)
        with open("./pkls/eval_token_pairs.pkl", "wb") as f:
            pickle.dump(token_pairs, f)

    lang_en = LimitedLang(source_lang, 50000)
    lang_ja = LimitedLang(target_lang, 50000)

    #print(lang_ja.word2count)

    if not valid:
        with open("./pkls/50k_en_words.pkl", mode='wb') as f:
            pickle.dump(lang_en, f)

        with open("./pkls/50k_jp_words.pkl", mode='wb') as f:
            pickle.dump(lang_ja, f)
    else:
        with open("./pkls/eval_50k_en_words.pkl", mode='wb') as f:
            pickle.dump(lang_en, f)

        with open("./pkls/eval_50k_jp_words.pkl", mode='wb') as f:
            pickle.dump(lang_ja, f)

    return source_lang, target_lang, pairs, token_pairs


def loadPkls():
    with open("./pkls/50k_en_words.pkl", "rb") as f:
        source_lang = pickle.load(f)
    with open("./pkls/50k_jp_words.pkl", "rb") as f:
        target_lang = pickle.load(f)
    with open("./pkls/token_pairs_preprop.pkl", "rb") as f:
    #with open("./pkls/eval_pairs_preprop.pkl", "rb") as f:
        token_pairs = pickle.load(f)

    return source_lang, target_lang, token_pairs

def loadEval():
    with open("./pkls/eval_pairs_preprop.pkl", "rb") as f:
        token_pairs = pickle.load(f)

    return token_pairs
