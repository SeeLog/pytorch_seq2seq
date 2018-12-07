import pickle
import random
from tqdm import tqdm

from Lang import Lang
from Lang import LimitedLang

from my_utils import MAX_LENGTH

"""
japan = Lang("ja")
print(japan.mecab_tokenize("ニューラルネットワークによってChatbotを制作します．"))
"""

with open("./pkls/train_token_pairs.pkl", mode="rb") as f:
    pairs = pickle.load(f)

n_pairs = []

for p in tqdm(pairs):
    if len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH:
        n_pairs.append(p)

with open("./pkls/token_pairs_preprop.pkl", mode="wb") as f:
    pickle.dump(n_pairs, f)

print("BEF:", len(pairs))
print("AFT:", len(n_pairs))




with open("./pkls/eval_token_pairs.pkl", mode="rb") as f:
    pairs = pickle.load(f)

n_pairs = []

for p in tqdm(pairs):
    if len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH:
        n_pairs.append(p)

with open("./pkls/eval_pairs_preprop.pkl", mode="wb") as f:
    pickle.dump(n_pairs, f)

print("BEF:", len(pairs))
print("AFT:", len(n_pairs))

exit()

with open("./pkls/Lang_en.pkl", mode='rb') as f:
    lang_en = pickle.load(f)

with open("./pkls/Lang_ja.pkl", mode='rb') as f:
    lang_ja = pickle.load(f)

#print(list(lang_ja.word2index.keys())[5], lang_ja.word2index[list(lang_ja.word2index.keys())[5]])
#print(list(lang_ja.word2count.keys())[5], lang_ja.word2count[list(lang_ja.word2count.keys())[5]])

lang_en = LimitedLang(lang_en, 50000)
lang_ja = LimitedLang(lang_ja, 50000)

#print(lang_ja.word2count)

with open("./pkls/50k_en_words.pkl", mode='wb') as f:
    pickle.dump(lang_en, f)

with open("./pkls/50k_jp_words.pkl", mode='wb') as f:
    pickle.dump(lang_ja, f)


"""
with open("./pkls/token_pairs.pkl", mode='rb') as f:
    pairs = pickle.load(f)

n_pairs = []

for i, pair in enumerate(tqdm(pairs)):
    print(pair)
    n_pair = [pair[0], pair[1]]
    n_pairs.append(n_pair)

print(n_pairs[16])
"""