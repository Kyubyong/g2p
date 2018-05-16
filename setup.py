#from distutils.core import setup
from setuptools import setup
setup(
  name = 'g2p_en',
  packages = ['g2p_en'], # this must be the same as the name above
  version = '1.0',
  description = 'A Simple Python Module for English Grapheme To Phoneme Conversion',
  author = 'Kyubyong Park & Jongseok Kim',
  author_email = 'kbpark.linguist@gmail.com',
  url = 'https://github.com/Kyubyong/g2p', # use the URL to the github repo
  download_url = 'https://github.com/Kyubyong/g2p/archive/1.0.tar.gz', # I'll explain this in a second
  keywords = ['g2p','g2p_en'], # arbitrary keywords
  classifiers = [],
  install_requires = [
    'numpy>=1.13.1',
    'tensorflow >= 1.3.0',
    'nltk>=3.2.4',
    'inflect>=0.3.1',
    'distance>=0.1.3',
    'future>=0.16.0'
  ],
  license='Apache Software License'
)

