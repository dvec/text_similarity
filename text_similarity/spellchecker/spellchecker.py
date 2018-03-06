from itertools import product

VOWELS = set('аоиеэыуюя')
ALPHABET = set('абвгдежзийклмнопрстуфхцчшщъыьэюя')


class SpellChecker:
    def __init__(self, dictionary, use_cache=True):
        self._dict = dictionary
        self._use_cache = use_cache

        if use_cache:
            self._cached = {}

    @staticmethod
    def _duplicates_count(string, idx):
        initial_idx = idx
        last = string[idx]
        while idx + 1 < len(string) and string[idx + 1] == last:
            idx += 1
        return idx - initial_idx

    @staticmethod
    def _hamming_distance(word1, word2):
        diffs = 0
        for ch1, ch2 in zip(word1, word2):
            if ch1 != ch2:
                diffs += 1
        return diffs

    def _frequency(self, word):
        return self._dict.get(word, 0)

    @staticmethod
    def _variants(word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in ALPHABET if b]
        inserts = [a + c + b for a, b in splits for c in ALPHABET]
        return set(deletes + transposes + replaces + inserts)

    @staticmethod
    def _swap_vowel(word):
        word = list(word)
        for idx, l in enumerate(word):
            if type(l) == list:
                pass
            elif l in VOWELS:
                word[idx] = list(VOWELS)

        for p in product(*word):
            yield ''.join(p)

    @classmethod
    def _double_variants(cls, word):
        return set(s for w in cls._variants(word) for s in cls._variants(w))

    @classmethod
    def _reductions(cls, word):
        word = list(word)
        for idx, l in enumerate(word):
            n = cls._duplicates_count(word, idx)
            if n:
                flat_dupes = [l * (r + 1) for r in range(n + 1)][:3]
                for _ in range(n):
                    word.pop(idx + 1)
                word[idx] = flat_dupes

        for p in product(*word):
            yield ''.join(p)

    @classmethod
    def _both(cls, word):
        for reduction in cls._reductions(word):
            for variant in cls._swap_vowel(reduction):
                yield variant

    def _suggestions(self, word):
        word = word.lower()

        return ({word} & self._dict.keys() or
                (set(self._reductions(word)) | set(self._swap_vowel(word)) |
                 set(self._variants(word)) | set(self._both(word)) |
                 set(self._double_variants(word))) & self._dict.keys() or set())

    def _best_n(self, word, n):
        assert n > 0
        sort = sorted(self._suggestions(word), key=lambda x: (self._hamming_distance(x, word), -self._frequency(x)))[:n]
        return sort or [word]

    def update_dict(self, new_dict):
        self._dict = new_dict
        self._cached = {}

    def correct(self, word):
        if self._use_cache and word in self._cached:
            return self._cached[word]

        result = self._best_n(word, 1)[0]
        if self._use_cache:
            self._cached[word] = result

        return result
