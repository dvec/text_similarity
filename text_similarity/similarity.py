import os
from math import log
from time import time

from gensim.models import Word2Vec
from logging import getLogger

from gensim.models.callbacks import CallbackAny2Vec

from text_similarity.spellchecker import SpellChecker
from text_similarity.utils import stemmer, prepare


class SimilarityAnalyzer:
    DEFAULT_WV_PATH = './.w2v/save'
    LOG = getLogger(__name__)

    @classmethod
    def load(cls, wv_path=DEFAULT_WV_PATH):
        begin = time()
        w2v = Word2Vec.load(wv_path)
        cls.LOG.info('Model is loaded in {} seconds'.format(time() - begin))
        return SimilarityAnalyzer(w2v, wv_path)

    @classmethod
    def new(cls, wv_path=DEFAULT_WV_PATH, **kwargs):
        cls.LOG.debug('Creating empty model...')
        w2v = Word2Vec(**kwargs)

        os.makedirs(os.path.join(*wv_path.split(os.sep)[:-1]), exist_ok=True)

        return SimilarityAnalyzer(w2v, wv_path)

    def __init__(self, w2v, save_wv):
        self._w2v = w2v
        self._save_wv = save_wv
        self._corpus_count = sum(x.count for x in self._w2v.wv.vocab.values())
        self._build_spell_checker()

    def _build_spell_checker(self):
        self._spell_checker = SpellChecker(dict(map(lambda x: (x[0], x[1].count), self._w2v.wv.vocab.items())))

    # Not the original tf-idf formula, may have some error
    def _tfidf(self, s, w):
        return (s.count(w) / len(s)) * log(self._corpus_count / self._w2v.wv.vocab[w].count)

    def _sentence_similarity(self, s1, s2):
        if s1 < s2:
            s1, s2 = s2, s1

        r = 0
        cnt = 0
        for i in s1:
            i = stemmer.stemWord(i)

            if i not in self._w2v.wv:
                continue
            cnt += 1

            m = 0
            for j in s2:
                j = stemmer.stemWord(j)
                if j in self._w2v.wv:
                    similarity = self._w2v.wv.similarity(i, j)
                    m = max(m, similarity)
            r += m * self._tfidf(s1, i)
        result = r / max(cnt, 1)
        return result

    @property
    def min_count(self):
        return self._w2v.vocabluary.min_count

    @min_count.setter
    def min_count(self, value):
        self._w2v.vocabluary.min_count = value

    def similarity(self, s1, s2):
        s1, s2 = map(lambda x: list(map(lambda x: list(map(self._spell_checker.correct, x)), prepare(x))), (s1, s2))
        if s1 < s2:
            s1, s2 = s2, s1

        r = 0
        for i in s1:
            m = 0
            for j in s2:
                m = max(m, self._sentence_similarity(i, j))
            r += m
        result = r / max(len(s1), 1)
        self.LOG.debug('Similarity for "{}" and "{}": {}'.format(s1, s2, result))
        return result

    def train(self, texts, save=True, epochs=1):
        def get_prepared_data():
            return (x for s in map(prepare, texts) for x in s)

        if len(self._w2v.wv.vectors):
            self._w2v.build_vocab(get_prepared_data(), update=True)
        else:
            self._w2v.build_vocab(get_prepared_data())
            if not len(self._w2v.wv.vectors):
                raise RuntimeError('Too small text. Try to set min_count to 1')

        self.save()

        if save:
            callbacks = [_EpochSaver(self)]
        else:
            callbacks = []

        total_examples = sum(map(lambda x: bool(len(x)), get_prepared_data()))
        self._w2v.train(get_prepared_data(), total_examples=total_examples, epochs=epochs, callbacks=callbacks)
        self._build_spell_checker()

    def save(self):
        self.LOG.debug('Saving model...')
        if self._save_wv:
            self._w2v.save(self._save_wv)


class _EpochSaver(CallbackAny2Vec):
    def __init__(self, analyzer):
        self._analyzer = analyzer

    def on_epoch_end(self, model):
        self._analyzer.save()
