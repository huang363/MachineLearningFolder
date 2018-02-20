import tensorflow as tf
import numpy as np

t = tf.constant([1, 2, 3,4,7,6])
water = tf.argmax(t)
answer = tf.gather(t,water)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(water))
    print(session.run(answer))