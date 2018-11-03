import numpy as np
import csv
from sklearn.utils import shuffle
import os
from sklearn import metrics

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

p = np.load('/home/vytran/public/tensorflow/z_iso.npy').astype(np.float32)
n = np.load('/home/vytran/public/tensorflow/qcd_iso.npy').astype(np.float32)

train_p = p[:504814]
train_n = n

# Stack data vertically in order to concatenate both positive and negative signal
train_data = np.vstack((train_p,train_n))
train_out = np.array([1]*len(train_p) + [0]*len(train_n))

numbertr = len(train_out)

# Shuffling
order = shuffle(range(numbertr),random_state=100)
train_out = train_out[order]
train_data = train_data[order,0::]

train_out = train_out.reshape((numbertr,1))
trainnb = 0.9

# Splitting between training set and cross-validation set
valid_data = train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out = train_out[int(trainnb*numbertr):numbertr]

train_data = train_data[0:int(trainnb*numbertr),0::]
train_data_out = train_out[0:int(trainnb*numbertr)]

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None,45])
y_ = tf.placeholder(tf.float32, shape=[None,1])

######## Model ###########
W1 = weight_variable([45,100])
b1 = bias_variable([100])
A1 = tf.nn.relu(tf.matmul(x, W1) + b1)
W2 = weight_variable( [100,1] )
b2 = bias_variable([1])
y = tf.matmul(A1,W2) + b2
##########################

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

ntrain = len(train_data)
print("ntrain is: ",ntrain)

batch_size = 141
num_batch = ntrain // batch_size
print(num_batch)

with tf.Session() as sess:

  # Start training

  sess.run(tf.global_variables_initializer())
  # We train 20 epoches
  for i in range(20):
      # Each epoch will have ~25000 batches
    cur_id = 0
    epoch_loss = 0
    for j in range(num_batch):
        batch_data = train_data[cur_id:cur_id+batch_size]
        batch_data_out = train_data_out[cur_id:cur_id+batch_size]
        cur_id = cur_id + batch_size
        _, batch_loss =  sess.run([train_step, cross_entropy], feed_dict={x: batch_data, y_:batch_data_out})
        epoch_loss += batch_loss
    print("Average epoch loss epoch {0}: {1}".format(i,epoch_loss/num_batch))


    ### Test the model using validation data set

  ntest = len(valid_data)

  _, loss = sess.run([train_step, cross_entropy], feed_dict={x: valid_data, y_:valid_data_out})
  prediction = tf.nn.sigmoid(y)
  pred = prediction.eval(feed_dict={x:valid_data, y_: valid_data_out})
  pred[pred > 0.5] = 1
  pred[pred < 0.5] = 0
  print("pred: ",pred)
  print("valid_data_out: ",valid_data_out)

  correct_pred = np.array(pred==valid_data_out)
  print(correct_pred[1:100])
  accuracy = np.sum(correct_pred.astype(int))
  
  print("total_correct_pred: ",accuracy)
  print("Number of test set: ",ntest)
  print("Accuracy: ", float(accuracy)/ntest)
  
          

























































