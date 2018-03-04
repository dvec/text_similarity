from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='text_similarity',
    version='0.2.2',
    install_requires=[
        'requests==2.9.1',
        'gensim==3.3.0',
        'PyStemmer==1.3.0',
        'stop-words==2015.2.23.1'
    ],
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
)
