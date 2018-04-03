#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:13:53 2017

@author: zach
"""

import sys
import tensorflow as tf
from tensorflow.python.platform import gfile

stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
reload(sys)
sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
sys.setdefaultencoding('utf-8')

with tf.Session() as sess:
    model_filename = "./combind_model/combind_model.pb"
    
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print sess.run(result)