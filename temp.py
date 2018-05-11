import urllib.request
import codecs
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# step1. construct cmu lookup dictionary
url = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"

cmu = dict()
lines = urllib.request.urlopen(url).read().decode('utf-8', errors="replace").splitlines()
for line in lines:
    if line.startswith(";;;"): continue # comments
    word, pron = line.split("  ")
    cmu[word] = pron

# step2. construct homograph dictionary
f = 'homographs.en'
homograph2features = dict()
for line in codecs.open(f, 'r', 'utf8').read().splitlines():
    if line.startswith("#"): continue # comment
    # print(line.strip().split("|"))
    headword, pron1, pron2, pos1 = line.strip().split("|")
    homograph2features[headword] = (pron1, pron2, pos1)

# step3. Load trained model
def word2pron(word):
    pron = ""
    pref, suf = word, ""
    while len(pref) > 0:
        print(pref, suf)

        if pref in cmu:
            # print(pref)
            pron += cmu[pref] + " "
            pref, suf = suf, ""
        else:
            pref, suf = pref[:-1], pref[-1] + suf
            if pref == "":
                pron += suf[0] + " "
                pref, suf = suf[1:], ""

    pron = pron.strip()
    if pron == "":
        return word
    else:
        return pron

word = "resistentialism".upper()
out = word2pron(word)
print(out)
# def token2pron(token):
#     word, pos = token
#     word = word.upper()
#
#     if word in homograph2features: # Check homograph
#         pron1, pron2, pos1 = homograph2features[word]
#         if pos.startswith(pos1):
#             pron = pron1
#         else:
#             pron = pron2
#     elif word in word2pron: # CMU dict
#         pron = word2pron(word)
#
#
#
#
#
# def g2p(text):
#     tokens = word_tokenize(text)
#     ret = []
#     for token in tokens:
#         pron = word2pron(token)
#         ret.extend(pron.split())
#         ret.append(" ")
#     ret = ret[:-1]
#
#
# AH0 F AO1 R D AH0 B AH0 L
#
# ["A", "I", ".", " ", "A"]
#
#
