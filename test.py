# -*- coding: utf-8 -*-

from __future__ import print_function

import g2p
import tensorflow as tf

text = u"I need your Résumé. She is my girl. He's my"
text = u"activationist "*9+u"activationist"
import timeit
print ("hi")
with g2p.Session():
    res = timeit.timeit('g2p.g2p(text)','from __main__ import text, g2p', number=10)
print(res)

#with g2p.Session():
#    res = g2p.g2p(text)

'''
# Check tensorflow graph
a = tf.zeros([10,100])
b = tf.layers.dense(a,100)
c = tf.layers.dense(b,100)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
res2 = sess.run(b)
saver.save(sess,'testlog/model'); print ("saved")
saver.restore(sess,'testlog/model'); print ("loaded")

from tensorflow.contrib.framework.python.framework import checkpoint_utils
var_list = checkpoint_utils.list_variables('testlog/model')
for v in var_list: print(v)
'''
