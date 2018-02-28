Text similarity analyzer for russian language
## Installation
```
pip install git+https://github.com/dvec/text_similarity.git
```
## Similarity
```python
from text_similarity import SimilarityAnalyzer
analyzer = SimilarityAnalyzer.load()
s1 = 'Сможем ли мы когда-либо достигнуть марса?'
s2 = 'Как ты считашь, ступит ли когда-либо нога человека на поверхность красной планеты?'
print(analyzer.similarity(s1, s2))
# 1.2891037479562908
# Bigger - better
```
## Categorization
```python
from text_similarity import SimilarityAnalyzer

categories = ['полететь на марс', 'заказать пиццу', 'погладить кота']
questions = ['Привет, мурзик', 'Это пиццерия?', 'Эй, Илон, где звездный человек?']

if __name__ == '__main__':
    analyzer = SimilarityAnalyzer.load()
    for s in questions:
        a = []
        for c in categories:
            a.append((analyzer.similarity(s, c), c))
        print(*map('{0[1]} - {0[0]}'.format, sorted(a, reverse=True)), sep='\n')
```
## Training
```python
from text_similarity import SimilarityAnalyzer, LibRuParser

if __name__ == '__main__':
    analyzer = SimilarityAnalyzer.empty()
    analyzer.train(LibRuParser(1000), update=False)
```
## Continue training
```python
from text_similarity import SimilarityAnalyzer, LibRuParser

if __name__ == '__main__':
    analyzer = SimilarityAnalyzer.load()
    analyzer.train(LibRuParser(1000))
```
