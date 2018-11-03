# Variables
import tensorflow as tf

x = tf.constant(35,name='x')
y = tf.Variable(x+5,name='y')

# Create a model, initialize variables
model = tf.global_variables_initializer()

# Run the model
with tf.Session() as session:
    session.run(model)
    print(session.run(y))

import numpy as np

x = tf.Variable(0,name='x')
result = tf.Variable(0,name='result')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        # Updating x
        x = x + 1
        print(session.run(x))

### TensorBoard
import tensorflow as tf

a = tf.add(1,2)
b = tf.multiply(a,3)
c = tf.add(4,5)
d = tf.multiply(c,6)
e = tf.multiply(4,5)
f = tf.div(c,6)
g = tf.add(b,d)
h = tf.multiply(g,f)

with tf.Session() as sess:
    print(sess.run(h))

# SummaryWrite at the end to create a folder containing information for TensorBoard to build the graph
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output/demo")
    print(sess.run(h))
    writer.add_graph(sess.graph)
    writer.close()















































