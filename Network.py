import tensorflow as tf
import Input as inputs
import random
import numpy as np



class Network():

    __rank = 0
    __error = 0
    __error_diff = 0
    __state = 'success'
    __w = None
    __b = None

    def __init__(self, w, b):

        self.init_w(w)
        self.init_b(b)

    def init_w(self, w):
        self.w = w

    def init_b(self, b):
        self.b = b

    def set_w(self, w):
        for i in range(2):
            #print(np.shape(w[i]))
            ini = tf.constant(w[i], shape=np.shape(w[i]))
            self.w[i] = tf.Variable(ini)

    def set_b(self, b):
        for i in range(np.shape(b)[0]):
            ini = tf.constant(b[i], shape=np.shape(b[i]))
            self.b[i] = tf.Variable(ini)

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def get_rank(self):
        return self.__rank

    def set_rank(self, rank):
        self.__rank = rank

    def get_error(self):
        return self.__error

    def set_error(self, error):
        self.__error = error

    def get_error_diff(self):
        return self.__error_diff

    def set_error_diff(self, error_diff):
        self.__error_diff = error_diff

    def get_state(self):
        return self.__state

    def set_state(self, state):
        self.__state = state

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


class Evolution():

    def __init__(self, population_size=2, generation=100):
        self.population_size = population_size
        self.generation = generation


    def population_optimizer(self, train_feature, train_label, test_feature, test_label, nets, step):
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
            init_error = tf.constant(sess.run(error, feed_dict={input: test_feature, output: test_label}))
            for j in range(step):
                sess.run(optimizer, feed_dict={input: train_feature, output: train_label})
            nets[i].set_error(sess.run(error, feed_dict={input: test_feature, output: test_label}))
            nets[i].set_error_diff(sess.run(init_error - error, feed_dict={input: test_feature, output: test_label}))
            print(nets[i].get_error_diff(),  nets[i].get_error(), sess.run(init_error, feed_dict={input: test_feature, output: test_label}))
            #print(sess.run(w))
            nets[i].set_w(sess.run(w))
            nets[i].set_b(sess.run(b))

        return nets

    def network_optimizer(self, train_feature, train_label, test_feature, test_label, net, step):
        input = tf.placeholder(tf.float32)
        output = tf.placeholder(tf.float32)
        w, b = net.get_all()
        l1 = tf.nn.tanh(tf.matmul(input, w[0]) + b[0])
        predict = tf.nn.tanh(tf.matmul(l1, w[1]) + b[1])
        error = tf.losses.mean_squared_error(predict, output)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(error)
        init_error = tf.constant(sess.run(error, feed_dict={input: test_feature, output: test_label}))
        print(sess.run(error, feed_dict={input: test_feature, output: test_label}))
        #print(sess.run(w))
        for j in range(step):
            sess.run(optimizer, feed_dict={input: train_feature, output: train_label})
        net_error = sess.run(error, feed_dict={input: test_feature, output: test_label})
        error_diff = sess.run(init_error - error, feed_dict={input: test_feature, output: test_label})
        print(sess.run(error, feed_dict={input: test_feature, output: test_label}))
        return sess.run(w), sess.run(b), net_error, error_diff

    def net_updated(self, net, w, b, net_error, error_diff):
        try:
            net.set_w(w)
            net.set_b(b)
            net.set_error(net_error)
            net.set_error_diff(error_diff)
        except:
            raise TypeError


    def state_marker(self, nets):
        for i in range(len(nets)):
            if nets[i].get_error() > 0.1 and nets[i].get_error_diff() < 0.15:
                nets[i].set_state('failure')
        return nets

    def rank(self, nets):
        rank_list = {}
        error_list = []
        for i in range(len(nets)):
            error_list.append(nets[i].get_error())
        error_list.sort()
        for i in range(len(nets)):
            rank_list[i] = error_list[i]
        for i in range(len(nets)):
            for key in rank_list:
                if nets[i].get_error() == rank_list[key]:
                    nets[i].set_rank(key + 1)
        return nets

    def sort_nets(self, nets):
        i = 0
        sortednets = []
        rank = 1
        size = 0
        while 1:
            if nets[i].get_rank() == rank:
                sortednets.append(nets[i])
                rank += 1
                i = 0
                size += 1
            else:
                i += 1
            if size == len(nets):
                break
        return sortednets











tra_feature, tra_label, test_feature, test_label = inputs.Input().data_generator()


population = Population(2, 3, 1).nets
population = Evolution(2).population_optimizer(tra_feature, tra_label, test_feature, test_label, population, 20)




# population = Evolution(1).rank(population)
# population = Evolution(1).sort_nets(population)

w, b, e, r = Evolution(2).network_optimizer(tra_feature, tra_label, test_feature, test_label, population[0], 20)
Evolution(1).net_updated(population[0], w, b, e, r)

w, b, e, r = Evolution(2).network_optimizer(tra_feature, tra_label, test_feature, test_label, population[1], 20)
Evolution(1).net_updated(population[1], w, b, e, r)

















# writer = tf.summary.FileWriter('./resource')
# writer.add_graph(sess.graph)


