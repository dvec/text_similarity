import cgi
import re
from text_simmilarity.utils import stemmer

import requests
from gensim.utils import simple_preprocess


class LibRuParser:
    PATTERN_1 = re.compile('<A HREF=([A-Z]+/)>')
    PATTERN_2 = re.compile('<A HREF=(\w+\.txt)>')
    PATTERN_3 = re.compile('[^а-яА-Я\s]')

    DEFAULT_MIRROR = 'http://kulichki.com/moshkow/'

    @staticmethod
    def _curl(url):
        r = requests.get(url)

        params = cgi.parse_header(r.headers.get('content-type'))[0]
        server_encoding = ('charset' in params) and params['charset'].strip("'\"") or None
        r.encoding = server_encoding or r.apparent_encoding

        return r.text

    def __init__(self, count, mirror=DEFAULT_MIRROR):
        self.mirror = mirror
        self.left = count

    def __iter__(self):
        matches = self.PATTERN_1.findall(self._curl(self.DEFAULT_MIRROR))

        for match in matches:
            full_url = self.DEFAULT_MIRROR + match
            new_matches = self.PATTERN_1.findall(self._curl(full_url))
            for new_match in new_matches:
                full_full_url = full_url + new_match
                for file in map(full_full_url.__add__, self.PATTERN_2.findall(self._curl(full_full_url))):
                    for i in self._prepare(self._curl(file)):
                        yield i
                        self.left -= 1
                        if self.left <= 0:
                            return

    def _prepare(self, text):
        text = self.PATTERN_3.sub('', text)
        lines = filter(bool, text.split('\n'))
        return map(simple_preprocess, [' '.join(stemmer.stemWords(l.split())) for l in lines])
