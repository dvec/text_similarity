from os import walk
from os.path import isdir, sep

from text_simmilarity.train.iparser import IParser


class FileParser(IParser):
    def __init__(self, files, **kwargs):
        """
        :param files: List of files to read or directories with files
        :param count: Count of files to read
        """
        super().__init__(**kwargs)
        self._files = files

    def find_files(self):
        for file in self._files:
            if isdir(file):
                for r, _, f in walk(file):
                    for f_ in f:
                        yield r + sep + f_
            else:
                yield file

    def _get_data(self):
        for file in self.find_files():
            with open(file) as f:
                yield f.read()
