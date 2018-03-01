from logging import basicConfig

from text_similarity import SimilarityAnalyzer

basicConfig(level=0)
categories = ['полететь на марс', 'заказать пиццу', 'погладить кота']

if __name__ == '__main__':
    analyzer = SimilarityAnalyzer.load()

    while 1:
        s = input()
        a = []
        for c in categories:
            a.append((analyzer.similarity(s, c), c))
        print(*map('{0[1]} - {0[0]}'.format, sorted(a, reverse=True)), sep='\n')
