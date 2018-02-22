from abc import ABCMeta, abstractmethod
from logging import getLogger
from math import inf


class IParser(metaclass=ABCMeta):
    LOG = getLogger(__name__)

    def __init__(self, *args, count=inf, **kwargs):
        self._left = count

    @abstractmethod
    def _get_data(self):
        raise NotImplementedError

    def __iter__(self):
        gen = self._get_data()

        while self._left:
            yield next(gen)
            self._left -= 1
            self.LOG.info('{} files left'.format(self._left))
