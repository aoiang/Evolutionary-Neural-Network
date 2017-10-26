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
    __nodes = [0] * 12
    __relu = []
    __elu = []

    def __init__(self, w, b, id):

        self.init_w(w)
        self.init_b(b)
        self.init_id(id)

    def init_w(self, w):
        self.w = w

    def init_b(self, b):
        self.b = b

    def init_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def set_elu(self, elu, op='append', copy=False):

        if copy:
            self.__elu.extend(elu)
        else:
            if op == 'append':
                self.__elu.append(elu)
            if op == 'remove':
                self.__elu.remove(elu)
        self.__elu = list(set(self.__elu))

    def set_relu(self, relu, op='append', copy=False):
        if copy:
            self.__elu.extend(relu)
        else:
            if op == 'append':
                self.__relu.append(relu)
            if op == 'remove':
                self.__relu.remove(relu)
        self.__relu = list(set(self.__relu))

    def get_relu(self):
        return self.__relu

    def get_elu(self):
        return self.__elu


    def set_w(self, w, init=False):
        self.__nodes = [0] * 12
        if not init:
            for i in range(len(w)):
                if i == 0:
                    self.__nodes[0] = np.shape(w[i])[0]
                ini = tf.constant(w[i], shape=np.shape(w[i]))
                self.w[i] = tf.Variable(ini)
                self.__nodes[i+1] = np.shape(w[i])[1]
        else:
            for i in range(len(w)):
                if i == 0:
                    self.__nodes[0] = np.shape(w[i])[0]
                ini = tf.random_uniform(shape=np.shape(w[i]))
                self.w[i] = tf.Variable(ini)
                self.__nodes[i + 1] = np.shape(w[i])[1]


    def set_b(self, b, init=False):
        if not init:
            for i in range(len(b)):
                ini = tf.constant(b[i], shape=np.shape(b[i]))
                self.b[i] = tf.Variable(ini)
        else:
            for i in range(len(b)):
                ini = tf.constant(0.1, shape=np.shape(b[i]))
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

    def get_nodesnum(self, layer=2, is_all=False):
        if is_all:
            return self.__nodes
        return self.__nodes[layer-1]

    def get_layer(self):
        return len(self.w)



class Population():

    def __init__(self, total_num, feature_number, output_num, layer_num=4, auto_generate=True, random_layer=True):
        self.total = total_num
        self.feature_number = feature_number
        self.output_num = output_num
        self.nets = [0] * total_num
        if auto_generate:
            for i in range(total_num):
                self.nets[i] = self.network_generator(i, layer_num, random_layer)

    def weight_init(self, shape):
        init = tf.random_uniform(shape)
        return tf.Variable(init)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def network_generator(self, id, layer_num=4, random_layer=False):
        if random_layer:
            layer_num = random.randint(3, 5)
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
        net = Network(w, b, id)
        return net


































# writer = tf.summary.FileWriter('./resource')
# writer.add_graph(sess.graph)


