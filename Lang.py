import re
import nagisa

from my_utils import __SOS__, __EOS__

class Lang():
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: __SOS__, 1: __EOS__}
        self.n_words = len(self.index2word)

    def addSentence(self, sentence):
        tokens = self.tokenize(sentence)
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
            return nagisa.tagging(sentence).words
