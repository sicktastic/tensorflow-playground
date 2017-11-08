import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

sess = tf.Session()
zeroD = tf.constant(5)
print(sess.run(tf.rank(zeroD)))

oneD = tf.constant(["How", "are", "you?"])
print(sess.run(tf.rank(oneD)))

twoD = tf.constant([[1.0, 2.3], [1.5, 2.9]])
print(sess.run(tf.rank(twoD)))

threeD = tf.constant([[[1,2], [3,4]], [[1,2],[3,4]]])
print(sess.run(tf.rank(threeD)))

sess.close()
