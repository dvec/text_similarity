from abc import ABCMeta, abstractmethod
from logging import getLogger
from math import inf


class IParser(metaclass=ABCMeta):
    LOG = getLogger(__name__)

    def __init__(self, count=inf, retry_count=10, retry_delay=10):
        self._count = count

        assert retry_count >= 0
        self._try_count = retry_count + 1
        assert retry_delay >= 0
        self._retry_delay = retry_delay

    @abstractmethod
    def _get_data(self):
        raise NotImplementedError

    def __iter__(self):
        gen = self._get_data()
        left = self._count

        while left > 0:
            yield next(gen)
            left -= 1
            self.LOG.info('{} files left'.format(left))
