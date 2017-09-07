import Data_preprocessing as dp
import Data_reader as reader
import tensorflow as tf

def get_data():
    data = reader.data_formatting(reader.data_extracter())
    feature, label = reader.feature_extracter(data, ['PATH_LTH', 'FREQUENCY', 'RR_01', 'A_01'], has_name=False)
    feature = reader.data_formatting(feature)
    label = reader.data_formatting(label)
    feature = (dp.fill_missing(feature, method='mean'))
    label = (dp.fill_missing(label, method='mean'))
    print(len(feature[0]), len(label[0]))
    return feature, label

def weight_init(shape):
    init = tf.random_uniform(shape)
    return tf.Variable(init)

def bias_variable(shape):
  initial = tf.constant(0.6, shape=shape)
  return tf.Variable(initial)


w1 = weight_init((86, 5))
w2 = weight_init((5, 8))
w3 = weight_init((8, 86))
w4 = weight_init((3, 1))

b1 = bias_variable((5,))
b2 = bias_variable((8,))
b3 = bias_variable((86,))
b4 = bias_variable((1,))

feature, label = get_data()

input = tf.constant(feature, dtype=tf.float32)
output = tf.constant(label, dtype=tf.float32)

l1 = tf.nn.relu(tf.matmul(input, w1) + b1)
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
predict_t = tf.nn.relu(tf.matmul(l2, w3) + b3)
predict_muti = tf.transpose(predict_t)
predict = tf.transpose(tf.nn.relu(tf.matmul(predict_muti, w4) + b4))


Si = predict / output
Vi = tf.log(Si)
meanV, varV = tf.nn.moments(Vi, axes=[1])


error = (tf.add(tf.square(meanV), varV)) ** 0.5


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(predict_t))
print(sess.run(output))
print(sess.run(error))


optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(error)
for i in range(2000):
    sess.run(optimizer)
    print(sess.run(predict))
print(sess.run(error))
print(sess.run(predict))