import tensorflow as tf
import Input as inputs
import random


class Network():

    def __init__(self, w, b):
        self.set_w(w)
        self.set_b(b)

    def set_w(self, w):
        self.w = w

    def set_b(self, b):
        self.b = b

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def get_all(self):
        return self.w, self.b


class Population():

    def __init__(self, total_num, feature_number, output_num, auto_generate=True):
        self.total = total_num
        self.feature_number = feature_number
        self.output_num = output_num
        self.nets = [0] * total_num
        if auto_generate:
            for i in range(total_num):
                self.nets[i] = self.network_generator()

    def weight_init(self, shape):
        init = tf.random_uniform(shape)
        return tf.Variable(init)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def network_generator(self, layer_num=3):
        node = [0] * (layer_num - 2)
        for i in range(layer_num - 2):
            node[i] = random.randint(2, 10)
        w = [0] * (layer_num - 1)
        b = [0] * (layer_num - 1)
        w[0] = self.weight_init((self.feature_number, node[0]))
        for i in range(1, layer_num - 2):
            w[i] = self.weight_init((node[i - 1], node[i]))
        w[layer_num - 2] = self.weight_init((node[-1], self.output_num))
        for i in range(layer_num - 2):
            b[i] = self.bias_variable((node[i],))
        b[-1] = self.bias_variable((self.output_num,))
        net = Network(w, b)
        return net


feature, label = inputs.Input().data_generator()

def optimizer(feature, label, nets):
    for i in range(len(nets)):
        input = tf.placeholder(tf.float32)
        output = tf.placeholder(tf.float32)
        w, b = nets[i].get_all()
        print(i)
        l1 = tf.nn.tanh(tf.matmul(input, w[0]) + b[0])
        predict = tf.nn.tanh(tf.matmul(l1, w[1]) + b[1])
        error = tf.losses.mean_squared_error(predict, output)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(error)
        for j in range(20000):
            sess.run(optimizer, feed_dict={input: feature[10:82], output: label[10:82]})
        print(sess.run(predict, feed_dict={input: feature[:10], output: label[:10]}), "kkkkkk\n")
        print(label[:10])
        nets[i].set_w(w)
        nets[i].set_b(b)
    return nets


optimizer(feature, label, Population(2, 3, 1).nets)












# writer = tf.summary.FileWriter('./resource')
# writer.add_graph(sess.graph)


