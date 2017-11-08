import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(6, name='constant_a')
b = tf.constant(3, name='constant_b')
c = tf.constant(10, name='constant_c')
d = tf.constant(5, name='constant_d')

mul = tf.multiply(a, b, name="mul")
div = tf.div(c, d, name="div")
addn = tf.add_n([mul, div], name="addn")

print(a)
print(mul)
print(div)
print(addn) 

sess = tf.Session()
print(sess.run(addn))

writer = tf.summary.FileWriter('./m2_example1', sess.graph)
writer.close()
sess.close()

#tensorboard --logdir="m2_example1"
