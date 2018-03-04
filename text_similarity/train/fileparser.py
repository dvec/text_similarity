from logging import getLogger
from os import walk
from os.path import sep
from time import sleep

from text_similarity.train.iparser import IParser


class FileParser(IParser):
    def __init__(self, directory, **kwargs):
        """
        :param files: Directory with files to parse
        :param count: Count of files to read
        """
        super().__init__(**kwargs)
        self._directory = directory

    def find_files(self):
        for r, _, f in walk(self._directory):
                for f_ in f:
                    yield r + sep + f_

    def _get_data(self):
        for file in self.find_files():
            for _ in range(self._try_count):
                try:
                    with open(file) as f:
                        yield f.read()
                except Exception as e:
                    getLogger(__name__).error(e)
                    sleep(self._retry_delay)
                else:
                    continue
