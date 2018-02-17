import os
from math import log

from gensim.models import Word2Vec

from text_simmilarity.train import LibRuParser
from text_simmilarity.utils import stemmer, prepare


class SimilarityAnalyzer:
    DEFAULT_W2V_PATH = './data/w2v/save'

    @classmethod
    def load(cls, w2v_path=DEFAULT_W2V_PATH):
        if os.path.exists(w2v_path):
            return SimilarityAnalyzer(Word2Vec.load(w2v_path), w2v_path)
        else:
            raise FileNotFoundError

    @classmethod
    def train(cls, count, epochs=1):
        parser = LibRuParser(count)
        w2v = Word2Vec(parser, iter=epochs)
        return SimilarityAnalyzer(w2v, cls.DEFAULT_W2V_PATH)

    def __init__(self, word2vec, save_w2v):
        self._w2v = word2vec
        self._save_w2v = save_w2v

    # Not the original tf-idf formula, may have some error
    def _tfidf(self, s, w):
        return (s.count(w) / len(s)) * log(self._w2v.corpus_count / self._w2v.wv.vocab[w].count)

    def _sentence_similarity(self, s1, s2):
        r = 0
        cnt = 0
        for i in s1:
            i = stemmer.stemWord(i)

            m = -float('inf')
            for j in s2:
                j = stemmer.stemWord(j)
                if i in self._w2v.wv and j in self._w2v.wv:
                    similarity = self._w2v.wv.similarity(i, j) * self._tfidf(s2, j)
                    m = max(m, similarity)
                    cnt += 1
            r += m
        return r / len(s1)

    def similarity(self, s1, s2):
        if len(s1.split()) > len(s2.split()):
            s2, s1 = s1, s2

        s1, s2 = sorted(map(prepare, (s1, s2)), key=len)
        r = 0
        for i in s1:
            m = -float('inf')
            for j in s2:
                m = max(m, self._sentence_similarity(i, j))
            r += m
        return r / len(s1)

    def save(self):
        if self._save_w2v:
            self._w2v.save(self._save_w2v)
