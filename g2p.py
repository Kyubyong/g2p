# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/g2p
'''
from __future__ import print_function

import tensorflow as tf
from train import Graph, hp, load_vocab
import numpy as np
import codecs
import re
import unicodedata
from expand import normalize_numbers

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict

import os

cmu = cmudict.dict()

# Load vocab
g2idx, idx2g, p2idx, idx2p = load_vocab()

# Load Graph

#try:
#    cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
#except:
#    cuda_devices = ""
#print ("cuda devices",cuda_devices)
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
g = tf.Graph()
with g.as_default():
    with tf.device('/cpu:0'):
        graph = Graph(); print("Graph Loaded")
        saver = tf.train.Saver()
config = tf.ConfigProto(
             device_count={'GPU' : 0},
             gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)#,
                                       #visible_device_list= '-1')

         )
#sess = tf.Session(graph=g, config=config)
#saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")
#os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

def predict(sess, word):
    '''
    Returns predicted pronunciation of `word` which does NOT exist in the dictionary.
    :param word: string.
    :return: pron: A list of phonemes
    '''

    graphemes = word + "E"  # EOS
    graphemes += "P" * hp.maxlen  # Padding

    x = [g2idx.get(g, 2) for g in graphemes[:hp.maxlen]]  # 2: <UNK>
    x = np.array(x, np.int32)
    x = np.expand_dims(x, 0) # (1, maxlen)


    ## Autoregressive inference
    preds = np.zeros((1, hp.maxlen), np.int32)
    for j in range(hp.maxlen):
        _preds = sess.run(graph.preds, {graph.x: x, graph.y: preds})
        preds[:, j] = _preds[:, j]

    # convert to string
    pron = [idx2p[idx] for idx in preds[0]]
    if "<EOS>" in pron:
        eos = pron.index("<EOS>")
        pron = pron[:eos]

    return pron

# Construct homograph dictionary
f = 'homographs.en'
homograph2features = dict()
for line in codecs.open(f, 'r', 'utf8').read().splitlines():
    if line.startswith("#"): continue # comment
    headword, pron1, pron2, pos1 = line.strip().split("|")
    homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)

def token2pron(sess, token):
    '''
    Returns pronunciation of word based on its pos.
    :param token: A tuple of (word, pos)
    :return: A list of phonemes
    '''
    word, pos = token

    if re.search("[a-z]", word) is None:
        pron = [word]

    elif word in homograph2features: # Check homograph
        pron1, pron2, pos1 = homograph2features[word]
        if pos.startswith(pos1):
            pron = pron1
        else:
            pron = pron2
    elif word in cmu: # CMU dict
        pron = cmu[word][0]
    else:
        pron = predict(sess,word)

    return pron

def g2p(text):
    '''
    Returns the pronunciation of text.
    :param text: A string. A sequence of words.
    :return: A list of phonemes.
    '''
    # normalization
    text = unicode(text)
    text = normalize_numbers(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()

    # tokenization
    words = word_tokenize(text)
    tokens = pos_tag(words) # tuples of (word, tag)

    # g2p
    #saver = tf.train.Saver()
    #config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session(graph=g, config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")
    #if True:
        ret = []
        for token in tokens:
            pron = token2pron(sess, token) # list of phonemes
            ret.extend(pron)
            ret.extend([" "])
        ret = ret[:-1]
        return ret

if __name__ == '__main__':
    text = u"I need your Résumé. She is my girl. He's my activationist activationist activationist."
    out = g2p(text)
    print(out)



# GPU : 7.83 (#3)
# CPU : 3.32 (#3)
# CPU : using Graph. 2.91 (#3)
# CPU : run session at first 1.51 (#3)
# GPU : run session at first 5.61 (#3)
