import re
from Stemmer import Stemmer

from stop_words import get_stop_words

stemmer = Stemmer('ru')


def prepare(text):
    text = re.sub('[^а-яА-Я\s]', '', text)
    result = [[]]

    for word in text.lower().split():
        result[-1].append(stemmer.stemWord(word))
        if word in get_stop_words('russian', cache=True):
            result.append([])

    return list(filter(bool, result))

