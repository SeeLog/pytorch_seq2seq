import re
#import nagisa
import MeCab

from my_utils import __SOS__, __EOS__

class Lang():
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: __SOS__, 1: __EOS__}
        self.n_words = len(self.index2word)
        self.tokenList = []
        self.mecab = MeCab.Tagger("-Owakati -d /home/shugo693/.local/lib/mecab/dic/ipadic")

    def addSentence(self, sentence):
        tokens = self.tokenize(sentence)
        self.tokenList.append(tokens)
        for token in tokens:
            self.addWord(token)

    def addWord(self, word):
        if word not in self.index2word:
            self.word2index[word] = word
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def tokenize(self, sentence):
        if self.name == 'en':
            return sentence.split(' ')
        elif self.name == 'ja':
            return self.mecab_tokenize(sentence)

    def mecab_tokenize(self, sentence):
        return self.mecab.parse(sentence).replace("\n", "").strip().split(" ")
