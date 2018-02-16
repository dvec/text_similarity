Text similarity analyzer for russian language
## Usage
```python
from text_simmilarity import SimilarityAnalyzer
analyzer = SimilarityAnalyzer.load()
s1 = 'Сможем ли мы когда-либо достигнуть марса?'
s2 = 'Как ты считашь, ступит ли когда-либо нога человека на поверхность красной планеты?'
print(analyzer.similarity(s1, s2))
# 0.6774318171107423
```
## TODO
Add tf-idf statistics
