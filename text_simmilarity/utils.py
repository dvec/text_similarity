from Stemmer import Stemmer

stemmer = Stemmer('ru')


def prepare_word(word):
    return stemmer.stemWord(word)
