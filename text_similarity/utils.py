import re
from Stemmer import Stemmer

from stop_words import get_stop_words

stemmer = Stemmer('ru')


def filter_text(text):
    return re.sub('[^а-я\s]', '', text.lower()).replace('.', ' . ')


def prepare(text, min_len=2):
    text = filter_text(text.replace('ё', 'е'))
    result = [[]]

    for word in text.split():
        word = stemmer.stemWord(word)
        if not word or len(word) < min_len:
            continue

        if word in get_stop_words('russian', cache=True):
            result.append([word])
        else:
            result[-1].append(word)

    return list(filter(bool, result))
