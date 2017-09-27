import Data_preprocessing as dp
import Data_reader as reader
import tensorflow as tf
import numpy as np

def get_data():
    data = reader.data_formatting(reader.data_extracter())
    feature, label = reader.feature_extracter(data, ['PATH_LTH', 'FREQUENCY', 'RR_01', 'A_01'], has_name=False)
    feature = reader.data_formatting(feature)
    label = reader.data_formatting(label)
    feature = (dp.fill_missing(feature, method='mean'))
    label = (dp.fill_missing(label, method='mean'))
    feature = dp.normalize(feature, is_max=True)
    label = dp.normalize(label, is_max=True)
    return np.transpose(np.array(feature)), np.transpose(np.array(label))


def weight_init(shape, names=''):
    init = tf.random_uniform(shape)
    return tf.Variable(init, name=names)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


w1 = weight_init((3, 20), names='w1')
w2 = weight_init((20, 1), names='w2')


b1 = bias_variable((20,))
b2 = bias_variable((1,))


feature, label = get_data()
print(feature)
#print(label)

# input = tf.placeholder(tf.float32)
# output = tf.placeholder(tf.float32)

input = tf.constant(feature, dtype=tf.float32, name='input')
output = tf.constant(label, dtype=tf.float32, name='output')

# test_in = tf.constant(label, dtype=tf.float32, name='output')[70:]
# test_out = tf.constant(label, dtype=tf.float32, name='output')[70:]





l1 = tf.nn.tanh(tf.matmul(input, w1) + b1, name='hadden_layer')
l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2, name='out_layer')




# predict_t = tf.nn.sigmoid(tf.matmul(l2, w3) + b3)
# predict_muti = tf.transpose(predict_t)
# predict = tf.transpose(tf.nn.sigmoid(tf.matmul(predict_muti, w4) + b4))


error = tf.losses.mean_squared_error(l2, output)

# Si = predict / output
# Vi = tf.log(Si)
# meanV, varV = tf.nn.moments(Vi, axes=[1])
#
#
# error = (tf.add(tf.square(meanV), varV)) ** 0.5


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# writer = tf.summary.FileWriter('./resource')
# writer.add_graph(sess.graph)

print(sess.run(tf.shape(output)))
# print(sess.run(error))


optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(error)
for i in range(10000):
    sess.run(optimizer)

    #print(sess.run(w2))


# sess.run(optimizer, feed_dict={input: feature[70:], output: label[70:]})
for i in range(86):
    print(sess.run(output[i]), sess.run(l2[i]))
# print(sess.run(output))
# print(sess.run(l2))


