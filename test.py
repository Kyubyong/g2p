# -*- coding: utf-8 -*-

from __future__ import print_function

from g2p import g2p
import tensorflow as tf

a = tf.zeros([10,100])
b = tf.layers.dense(a,100)
c = tf.layers.dense(b,100)

text = u"I need your Résumé. She is my girl. He's my activationist activationist activationist."
import timeit
print ("hi")
res = timeit.timeit('g2p(text)','from __main__ import text, g2p', number=3)
print(res)
#var_list = [v for v in tf.global_variables() ]
#print (var_list)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
res2 = sess.run(b)
saver.save(sess,'testlog/model'); print ("saved")
saver.restore(sess,'testlog/model'); print ("loaded")

from tensorflow.contrib.framework.python.framework import checkpoint_utils
var_list = checkpoint_utils.list_variables('testlog/model')
for v in var_list: print(v)
