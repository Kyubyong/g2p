import urllib.request
import codecs
import re

url = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"

lines = urllib.request.urlopen(url).read().decode('utf-8', errors="replace").splitlines()

word2prons = dict()
for line in lines:
    if line.startswith(";;;"): continue # comments
    word, pron = line.split("  ")
    word = re.sub("\(\d+\)", "", word) # e.g. PROJECT(1) -> PROJECT
    pron = pron.replace(" ", ".") # e.g. P OW1 L -> P.OW1.L
    if word in word2prons:
        word2prons[word].append(pron)
    else:
        word2prons[word] = [pron]

# print(word2prons)

homographs = [line.strip().split("/")[0].upper() for line in codecs.open('homographs.txt', 'r', 'utf8').read().splitlines() if len(line)>0]
# print(homographs)

for h in homographs:
    if h in word2prons:
        print(h, "|".join(word2prons[h]))




