import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from text_simmilarity.train import LibRuParser
from text_simmilarity.utils import stemmer


class SimilarityAnalyzer:
    DEFAULT_PATH = '../data/save/save'

    @classmethod
    def load(cls, path=None, save=None):
        if path is None:
            path = cls.DEFAULT_PATH

        if save is None:
            save = path

        if os.path.exists(path):
            return SimilarityAnalyzer(Word2Vec.load(path), save=save)
        else:
            raise FileNotFoundError

    def __init__(self, word2vec, save=None):
        if save is None:
            save = self.DEFAULT_PATH

        self._model = word2vec
        self._save = save

    def similarity(self, s1, s2):
        s1, s2 = sorted(map(simple_preprocess, (s1, s2)), key=len)
        print(s1, s2)

        r = 0
        for i in s1:
            i = stemmer.stemWord(i)

            m = -float('inf')
            for j in s2:
                j = stemmer.stemWord(j)
                if i in self._model.wv and j in self._model.wv:
                    similarity = self._model.wv.similarity(i, j)
                elif i == j:
                    similarity = 1
                else:
                    similarity = 0
                m = max(m, similarity)
            r += m
        return r / len(s1)

    def save(self):
        self._model.save(self._save)

    def train(self, count, epochs=1, auto_save=True):
        parser = LibRuParser(count)

        self._model.build_vocab(parser)
        for _ in range(epochs):
            self._model.train(parser)
            if auto_save:
                self.save()
