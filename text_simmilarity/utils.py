import re
from Stemmer import Stemmer

from stop_words import get_stop_words

stemmer = Stemmer('ru')


def prepare(text, min_len=2):
    text = re.sub('[^а-яА-Я\s]', '', text).replace('.', ' . ')
    result = [[]]

    for word in text.lower().split():
        word = stemmer.stemWord(word)
        if not word or len(word) < min_len:
            continue

        if word in get_stop_words('russian', cache=True):
            result.append([word])
        else:
            result[-1].append(word)

    return list(filter(bool, result))
