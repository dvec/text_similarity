from abc import ABCMeta, abstractmethod
from logging import getLogger
from math import inf


class IParser(metaclass=ABCMeta):
    LOG = getLogger(__name__)

    def __init__(self, *args, count=inf, **kwargs):
        self._count = count

    @abstractmethod
    def _get_data(self):
        raise NotImplementedError

    def __iter__(self):
        print('__iter__ call')
        gen = self._get_data()
        left = self._count

        while left > 0:
            yield next(gen)
            left -= 1
            self.LOG.info('{} files left'.format(left))
