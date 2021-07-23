import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

x = np.random.rand(200).astype(np.float32)
y = x*0.5+0.8
plt.plot(x,y,color = 'red')
weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

func = weights*x + biases

loss = tf.reduce_mean(tf.square(func - y)) # MSE

optimization = tf.train.GradientDescentOptimizer(0.5)
train = optimization.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(201):
    sess.run(train)
    if i % 10 == 0:
        plt.scatter(x,x*sess.run(weights)+sess.run(biases))
        print(i,sess.run(weights),sess.run(biases))
plt.show()
