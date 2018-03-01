from logging import basicConfig, INFO

from text_similarity import SimilarityAnalyzer, LibRuParser

EXAMPLES_COUNT = 1
EPOCHS = 2

basicConfig(level=INFO)

if __name__ == '__main__':
    parser = LibRuParser(count=EXAMPLES_COUNT)

    analyzer = SimilarityAnalyzer.empty()
    analyzer.train(LibRuParser(count=EXAMPLES_COUNT), epochs=EPOCHS // 2)

    del analyzer

    analyzer = SimilarityAnalyzer.load()
    analyzer.train(LibRuParser(count=EXAMPLES_COUNT), epochs=EPOCHS // 2)