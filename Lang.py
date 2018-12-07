import re
#import nagisa
import MeCab

from my_utils import __PAD__, __SOS__, __EOS__, __UNK__, __PAD_ID__, __SOS_ID__, __EOS_ID__, __UNK_ID__

class LimitedLang():
    """
    単語数を制限したLang
    """

    def __init__(self, lang_obj, maxcount):
        self.name = lang_obj.name
        self.word2index = {__PAD__: __PAD_ID__, __SOS__: __SOS_ID__, __EOS__: __EOS_ID__, __UNK__: __UNK_ID__}
        self.word2count = {}
        self.index2word = {__PAD_ID__: __PAD__, __SOS_ID__: __SOS__, __EOS_ID__: __EOS__, __UNK_ID__: __UNK__}
        self.n_words = len(self.index2word)

        sort_words = sorted(lang_obj.word2count.items(), key=lambda x: -x[1])
        for i, w2c in enumerate(sort_words):
            if i >= maxcount:
                break
            word = w2c[0]
            count = w2c[1]
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.word2count[word] = count



class Lang():
    def __init__(self, name):
        self.name = name
        self.word2index = {__PAD__: __PAD_ID__, __SOS__: __SOS_ID__, __EOS__: __EOS_ID__, __UNK__: __UNK_ID__}
        self.word2count = {}
        self.index2word = {__PAD_ID__: __PAD__, __SOS_ID__: __SOS__, __EOS_ID__: __EOS__, __UNK_ID__: __UNK__}
        self.n_words = len(self.index2word)
        self.tokenList = []
        self.mecab = MeCab.Tagger("-Owakati -d /home/shugo693/.local/lib/mecab/dic/ipadic")

    def addSentence(self, sentence):
        tokens = self.tokenize(sentence)
        self.tokenList.append(tokens)
        for token in tokens:
            self.addWord(token)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def tokenize(self, sentence):
        sentence = sentence.strip()
        if self.name == 'en':
            return sentence.split(' ')
        elif self.name == 'ja':
            return self.mecab_tokenize(sentence)

    def mecab_tokenize(self, sentence):
        return self.mecab.parse(sentence).replace("\n", "").strip().split(" ")
