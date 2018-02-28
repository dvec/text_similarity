from os import walk
from os.path import isdir, sep

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
            with open(file) as f:
                yield f.read()
