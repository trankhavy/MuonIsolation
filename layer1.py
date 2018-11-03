import numpy as np
import csv
# provide a way to consistently shuffle the data
from sklearn.utils import shuffle
# OS module can be used to interface with the operating system in which Python is using
import os
# The sklearn.metrics module includes score function, perfomance metrics and pairwise metrics and
# distance computation
from sklearn import metrics

def weight_variable(shape):
  initial = tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1,shape=shape)
  return tf.Variable(initial)

p = np.load('/home/vytran/public/tensorflow/z_iso.npy').astype(np.float32) # positive samples (zjets)
n = np.load('/home/vytran/public/tensorflow/qcd_iso.npy').astype(np.float32) # negative samples (qcd)

train_p = p[:504814]
train_n = n

train_data = np.vstack((train_p,train_n))
train_out = np.array([1]*len(train_p) + [0]*len(train_n))

numbertr = len(train_out)

# Shuffling
# Shuffling here use shuffle() function from sklearn.utils with random_state = seed
order = shuffle(range(numbertr),random_state=100)
train_out = train_out[order]
train_data = train_data[order,:]

train_out = train_out.reshape((numbertr,1))
trainb = 0.9 # Fraction used for training

# Splitting between traing set and validation set
valid_data = train_data[int(trainb*numbertr):numbertr,:]
valid_data_out = train_out[int(trainb*numbertr):numbertr]

train_data_out = train_out[0:int(trainb*numbertr)]
train_data = train_data[0:int(trainb*numbertr),:]

import tensorflow as tf
sess = tf.interactivesession()

x = tf.placeholder(tf.float32, shape=[None,45])
y_ = tf.placeholder(tf.float32, shape=[None,1])

######## Model ############
# Deciding on 100 neurons
W1 = weight_variable([45,100])
b1 = bias_variable([100])

A1 = tf.nn.relu(tf.matmul(x,W1) + b1) 
W2 = weight_variable([100,1])
b2 = bias_variable([1])
# The output of layer 1
y = tf.matmul(A1,W2) + b2
########################### 

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))
print("The cross-entropy is: ", cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

ntrain = len(train_data)

print("ntrain = ", ntrain)

batch_size = 141
epoch_size = ntrain // batch_size
cur_id = 0

saver = tf.train.Saver()

model_output_name = "layer1_100"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if os.path.exists('models/'+model_output_name+'/model_out.meta'):
        print("Model file exists already!")
        saver.restore(sess, 'models/'+model_output_name+'/model_out')
    else:
        for i in range(20):
            for j in range(epoch_size):
                batch_data = train_data[cur_id:cur_id+batch_size]
                batch_data_out = train_data_out[cur_id:cur_id+batch_size]
                cur_id = cur_id + batch_size
                train_step.run(feed_dict={x: batch_data, y_: batch_data_out})

        saver.save(sess, 'models/'+model_output_name + '/model_out')
        print("Model saved!")

    prediction = tf.nn.sigmoid(y)
    pred = prediction.eval(feed_dict={x: valid_data, y_: valid_data_out})
    print(pred)

    y = []
    signal_output = []
    background_output = []

# save output for later use
# signal_output and background_output is the valid_data_out and the prediction we made
  with open('models/layer1_100/signal.csv','w') as f:
      writer = csv.writer(f, delimiter=" ")
      for i in range(len(valid_data)):
          x = valid_data_out[i] # validation label
          if x == 1: # signal output
              signal_output.append(pred[i])
              y = pred[i]
              writer.writerows(zip(y,x))

  with open('models/layer1_100/background.csv','w') as f:
      writer = csv.writer(f, delimiter=" ")
      for i in range(len(valid_data)):
          x = valid_data_out[i] # validation level
          if x == 0:
              background_output.append([pred[i])
              y = pred[i]
              writer.writerows(zip(y,x))

  # Convert to numpy array
  s_output = np.array(signal_output)
  b_output = np.array(background_output)
  threshold = 0.5 # test
  s = s_output > threshold
  b = b_output > theshold

  # Signal count 
  ns_sel = len(s_output[s]) # count only elements larger than threshold
  ns_total = len(signal_output) 

  # Background count
  nb_sel = len(b_output[b]) 
  nb_total = len(background_output)

  # True positive rate
  print("signal: ", ns_sel , "/" , ns_total)
  # True negative rate
  print("background: ", nb_sel, "/", nb_total)

  # efficiency
  sig_eff = float(ns_sel)/float(ns_total)
  bkg_eff = float(nb_sel)/float(nb_total)

  print("signal eff = ", sig_eff, " background eff = ", bkg_eff)






































