import numpy as np
import csv
from sklearn.utils import shuffle
import os

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

p=np.load('/home/vytran/public/tensorflow/z_iso.npy').astype(np.float32) #positive samples (zjets)
n=np.load('/home/vytran/public/tensorflow/qcd_iso.npy').astype(np.float32) #negative samples (qcd)
train_p=p[:504814]
train_n=n

#train_p=np.load('materials/z_iso.npy').astype(np.float32) #positive samples (zjets)
#train_n=np.load('materials/qcd_iso.npy').astype(np.float32) #negative samples (qcd)

train_data=np.vstack((train_p,train_n))
train_out=np.array([1]*len(train_p)+[0]*len(train_n))

numbertr=len(train_out)

#Shuffling
order=shuffle(range(numbertr),random_state=100)
train_out=train_out[order]
train_data=train_data[order,0::]

train_out = train_out.reshape( (numbertr, 1) )
trainnb=0.9 # Fraction used for training

#Splitting between training set and cross-validation set
valid_data=train_data[int(trainnb*numbertr):numbertr,0::]
valid_data_out=train_out[int(trainnb*numbertr):numbertr]

train_data_out=train_out[0:int(trainnb*numbertr)]
train_data=train_data[0:int(trainnb*numbertr),0::]

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 45]) #20
y_ = tf.placeholder(tf.float32, shape=[None, 1])

##### Model #####
W1 = weight_variable( [45,100] )
b1 = bias_variable( [100] )
A1 = tf.nn.relu(tf.matmul(x, W1) + b1)
W2 = weight_variable( [100,100] )
b2 = bias_variable( [100] )
A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)
W3 = weight_variable( [100,100] )
b3 = bias_variable( [100] )
A3 = tf.nn.relu(tf.matmul(A2, W3) + b3)
W4 = weight_variable( [100,100] )
b4 = bias_variable( [100] )
A4 = tf.nn.relu(tf.matmul(A3, W4) + b4)
W5 = weight_variable( [100,100] )
b5 = bias_variable( [100] )
A5 = tf.nn.relu(tf.matmul(A4, W5) + b5)
W6 = weight_variable( [100,100] )
b6 = bias_variable( [100] )
A6 = tf.nn.relu(tf.matmul(A5, W6) + b6)
W7 = weight_variable( [100,100] )
b7 = bias_variable( [100] )
A7 = tf.nn.relu(tf.matmul(A6, W7) + b7)
W8 = weight_variable( [100,100] )
b8 = bias_variable( [100] )
A8 = tf.nn.relu(tf.matmul(A7, W8) + b8)
W9 = weight_variable( [100,1] )
b9 = bias_variable( [1] )
y = tf.matmul(A8,W9) + b9
##################

cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

ntrain = len(train_data)

print ("ntrain = ", ntrain)

batch_size = 105
epoch_size = ntrain // batch_size
cur_id = 0

saver = tf.train.Saver()

model_output_name = "layer8_100"

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  if os.path.exists('models/'+model_output_name+'/model_out.meta'):
      print ("Model file exists already!")
      saver.restore(sess, 'models/'+model_output_name+'/model_out')
  else:
    for i in range(20):
      for j in range(epoch_size):
        batch_data = train_data[cur_id:cur_id+batch_size]
        batch_data_out = train_data_out[cur_id:cur_id+batch_size]
        cur_id = cur_id+batch_size
        train_step.run(feed_dict={x: batch_data, y_: batch_data_out})

    saver.save(sess,  'models/'+model_output_name+'/model_out')
    print ("Model saved!")

  prediction = tf.nn.sigmoid(y) 
  pred = prediction.eval( feed_dict={x: valid_data, y_: valid_data_out} )
  print (pred)  

  y = []
  signal_output = []
  background_output = []

  #save output for later use
  with open('models/layer8_100/signal.csv', 'w') as f:
    writer = csv.writer(f, delimiter=" ")
    for i in range(len(valid_data)):
      x = valid_data_out[i] #valdiation label
      if x == 1: #signal output 
        signal_output.append( pred[i] ) 
        y = pred[i]
        writer.writerows( zip(y,x) )       

  with open('models/layer8_100/background.csv', 'w') as f:
    writer = csv.writer(f, delimiter=" ")
    for i in range(len(valid_data)):
      x = valid_data_out[i] #validation level
      if x == 0: #background output
        background_output.append( pred[i] ) 
        y = pred[i]
        writer.writerows( zip(y,x) )

  #convert to numpy array
  s_output = np.array(signal_output)
  b_output = np.array(background_output)
  threshold = 0.5 #test
  s = s_output > threshold
  b = b_output > threshold

  #signal count
  ns_sel = len(s_output[s]) # count only elements larger than threshold
  ns_total = len(signal_output) 
  
  #background count 
  nb_sel = len(b_output[b]) # count only elements larger than threshold
  nb_total = len(background_output)

  print ("signal : " , ns_sel ,  "/" , ns_total)
  print ("background : ", nb_sel ,  "/" , nb_total)
 
  #efficiency
  sig_eff = float(ns_sel)/float(ns_total) 
  bkg_eff = float(nb_sel)/float(nb_total)
 
  print ("signal eff = ", sig_eff, " background eff = ", bkg_eff)


