from time import time

from text_simmilarity.simmilarity import SimilarityAnalyzer

categories = ['полететь на марс', 'заказать пиццу', 'погладить кота']

if __name__ == '__main__':
    start = time()
    analyzer = SimilarityAnalyzer.load()
    print('LOADED {} at sec'.format(time() - start))
    while 1:
        s = input()
        a = []
        for c in categories:
            a.append((analyzer.similarity(s, c), c))
        print(*map('{0[1]} - {0[0]}'.format, sorted(a, reverse=True)), sep='\n')
