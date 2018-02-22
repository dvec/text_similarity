import os
from math import log
from time import time

from gensim.models import Word2Vec
from logging import getLogger

from gensim.models.callbacks import CallbackAny2Vec

from text_simmilarity.utils import stemmer, prepare


class SimilarityAnalyzer:
    DEFAULT_WV_PATH = './data/w2v/save'
    LOG = getLogger(__name__)

    @classmethod
    def load(cls, wv_path=DEFAULT_WV_PATH):
        if os.path.exists(wv_path):
            begin = time()
            w2v = Word2Vec.load(wv_path)
            cls.LOG.info('Model is loaded in {} seconds'.format(time() - begin))
            return SimilarityAnalyzer(w2v, wv_path)
        else:
            raise FileNotFoundError

    @classmethod
    def empty(cls):
        cls.LOG.debug('Creating empty model...')
        w2v = Word2Vec()
        return SimilarityAnalyzer(w2v, cls.DEFAULT_WV_PATH)

    def __init__(self, w2v, save_wv):
        self._w2v = w2v
        self._save_wv = save_wv
        self._corpus_count = sum(x.count for x in self._w2v.wv.vocab.values())

    # Not the original tf-idf formula, may have some error
    def _tfidf(self, s, w):
        return (s.count(w) / len(s)) * log(self._corpus_count / self._w2v.wv.vocab[w].count)

    def _sentence_similarity(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        r = 0
        cnt = 0
        for i in s1:
            i = stemmer.stemWord(i)

            if i not in self._w2v.wv:
                continue
            cnt += 1

            m = -float('inf')
            for j in s2:
                j = stemmer.stemWord(j)
                if j in self._w2v.wv:
                    similarity = self._w2v.wv.similarity(i, j) * self._tfidf(s2, j)
                    m = max(m, similarity)
            r += m
        result = r / max(cnt, 1)
        return result

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
        result = r / max(len(s1), 1)
        self.LOG.debug('Similarity for "{}" and "{}": {}'.format(s1, s2, result))
        return result

    def train(self, texts, save=True, update=True, epochs=1):
        iterator = map(prepare, texts)
        self._w2v.build_vocab(iterator, update=update)
        self.save()

        if save:
            callbacks = [_EpochSaver(self)]
        else:
            callbacks = []

        self._w2v.train(iterator, total_examples=self._w2v.corpus_count, epochs=epochs, callbacks=callbacks)

    def save(self):
        self.LOG.debug('Saving model...')
        if self._save_wv:
            self._w2v.save(self._save_wv)


class _EpochSaver(CallbackAny2Vec):
    def __init__(self, analyzer):
        self._analyzer = analyzer

    def on_epoch_end(self, model):
        self._analyzer.save()
