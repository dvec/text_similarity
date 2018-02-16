from text_simmilarity.simmilarity import SimilarityAnalyzer

if __name__ == '__main__':
    analyzer = SimilarityAnalyzer.load()
    while 1:
        s1 = input()
        s2 = input()
        print(analyzer.similarity(s1, s2))
