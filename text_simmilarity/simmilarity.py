import os
from math import log

from gensim.models import Word2Vec, KeyedVectors

from text_simmilarity.train import LibRuParser
from text_simmilarity.utils import stemmer, prepare


class SimilarityAnalyzer:
    DEFAULT_WV_PATH = './data/w2v.bin'

    @classmethod
    def load(cls, wv_path=DEFAULT_WV_PATH):
        if os.path.exists(wv_path):
            return SimilarityAnalyzer(KeyedVectors.load_word2vec_format(wv_path), wv_path)
        else:
            raise FileNotFoundError

    @classmethod
    def train(cls, count, epochs=1):
        parser = LibRuParser(count)
        w2v = Word2Vec(parser, iter=epochs)
        return SimilarityAnalyzer(w2v.wv, cls.DEFAULT_WV_PATH)

    def __init__(self, wv, save_wv):
        self._wv = wv
        self._save_wv = save_wv
        self._corpus_count = sum(x.count for x in self._wv.vocab.values())

    # Not the original tf-idf formula, may have some error
    def _tfidf(self, s, w):
        return (s.count(w) / len(s)) * log(self._corpus_count / self._wv.vocab[w].count)

    def _sentence_similarity(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        r = 0
        cnt = 0
        for i in s1:
            i = stemmer.stemWord(i)

            if i not in self._wv:
                continue
            cnt += 1

            m = -float('inf')
            for j in s2:
                j = stemmer.stemWord(j)
                if j in self._wv:
                    similarity = self._wv.similarity(i, j) * self._tfidf(s2, j)
                    m = max(m, similarity)
            r += m
        return r / max(cnt, 1)

    def similarity(self, s1, s2):
        s1, s2 = sorted((prepare(x) for x in (s1, s2)), key=len)

        if len(s1) > len(s2):
            s1, s2 = s2, s1

        r = 0
        for i in s1:
            m = -float('inf')
            for j in s2:
                m = max(m, self._sentence_similarity(i, j))
            r += m
        return r / max(len(s1), 1)

    def save(self):
        if self._save_wv:
            self._wv.save_word2vec_format(self._save_wv)
