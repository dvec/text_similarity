import cgi
import re
from logging import getLogger
from time import sleep

import requests

from text_similarity.train.iparser import IParser


class LibRuParser(IParser):
    PATTERN_1 = re.compile('<A HREF=([A-Z]+/)>')
    PATTERN_2 = re.compile('<A HREF=(\w+\.txt)>')
    PATTERN_3 = re.compile('[^а-яА-Я\s]')

    DEFAULT_MIRROR = 'http://kulichki.com/moshkow/'

    def __init__(self, mirror=DEFAULT_MIRROR, **kwargs):
        self._mirror = mirror
        super().__init__(**kwargs)

    def _curl(self, url):
        for _ in range(self._try_count):
            try:
                r = requests.get(url)
                params = cgi.parse_header(r.headers.get('content-type'))[0]
                server_encoding = ('charset' in params) and params['charset'].strip("'\"") or None
                r.encoding = server_encoding or r.apparent_encoding
            except IOError as e:
                getLogger(__name__).error(e)
                sleep(self._retry_delay)
            else:
                return r.text

        return ''

    def _get_data(self):
        matches = self.PATTERN_1.findall(self._curl(self.DEFAULT_MIRROR))

        for match in matches:
            full_url = self.DEFAULT_MIRROR + match
            new_matches = self.PATTERN_1.findall(self._curl(full_url))
            for new_match in new_matches:
                full_full_url = full_url + new_match
                for file in map(full_full_url.__add__, self.PATTERN_2.findall(self._curl(full_full_url))):
                    yield self._curl(file)
