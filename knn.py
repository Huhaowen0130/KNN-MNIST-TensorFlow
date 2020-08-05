# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:49:17 2020

@author: 63554
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy import stats
import time

# Hyper-parameters
NumOfTrain = 55000
NumOfTest = 10000
k = 3

# Load Regular MNIST(28x28)
mnist = input_data.read_data_sets('mnist/', one_hot=True)
x_train, y_train = mnist.train.next_batch(NumOfTrain)
x_test, y_test = mnist.test.next_batch(NumOfTest)

target = tf.placeholder(tf.float32, [1, 784])
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None, 10])
euclid_dist = tf.sqrt(tf.reduce_sum(tf.square(target - x), 1))

model = tf.nn.top_k(tf.negative(euclid_dist), k)

acc_list = []
error_list1 = []  # the true classes of misclassified images
error_list2 = []  # the predicted classes of misclassified images
cnt = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    start = time.time()
    for i in range(x_test.shape[0]):
        ind = sess.run(model, feed_dict={x:x_train, y:y_train, target:np.asmatrix(x_test[i])})

        pred = []        
        for j in range(k):
            pred.append(np.argmax(y_train[ind[1][j]]))
            
        pred_class = stats.mode(pred).mode[0]
        true_class = np.argmax(y_test[i])
        if pred_class == true_class:
            acc_list.append(1)
        else:
            acc_list.append(0)
            error_list1.append(true_class)
            error_list2.append(pred_class)
            cnt += 1
            print("----------The %dth image is misclassified, true_class: %d, pred_class: %d, the %dth misclassification" % (i, true_class, pred_class, cnt))
            
print("----------accuracy = {:.3f}".format(np.mean(acc_list)))