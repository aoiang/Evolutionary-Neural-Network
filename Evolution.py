import tensorflow as tf
import Input as inputs
import random
import numpy as np
from Network import Population, Network
from functools import reduce



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
            l = [0] * len(w)
            for j in range(len(w)):
                if j == 0:
                    l[j] = tf.nn.tanh(tf.matmul(input, w[j]) + b[j])
                if j > 0 and j < len(w) - 1:
                    l[j] = tf.nn.tanh(tf.matmul(l[j-1], w[j]) + b[j])
                if j == len(w) - 1:
                    predict = tf.nn.tanh(tf.matmul(l[j-1], w[j]) + b[j])
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
            nets[i].set_w(sess.run(w))
            nets[i].set_b(sess.run(b))

        return nets

    def network_optimizer(self, train_feature, train_label, test_feature, test_label, net, step, final=False):
        input = tf.placeholder(tf.float32)
        output = tf.placeholder(tf.float32)
        #print(net)
        w, b = net.get_all()
        l = [0] * len(w)
        for j in range(len(w)):
            #print(j,'dfsfadgdasg')
            if j == 0:
                if j in net.get_relu():
                    if final:
                        print('goes relu!!!')
                    l[j] = tf.nn.relu(tf.matmul(input, w[j]) + b[j])
                elif j in net.get_elu():
                    l[j] = tf.nn.elu(tf.matmul(input, w[j]) + b[j])
                else:
                    l[j] = tf.nn.tanh(tf.matmul(input, w[j]) + b[j])
            if j > 0 and j < len(w) - 1:
                if j in net.get_relu():
                    if final:
                        print('goes relu!!!')
                    l[j] = tf.nn.relu(tf.matmul(l[j - 1], w[j]) + b[j])
                elif j in net.get_elu():
                    l[j] = tf.nn.elu(tf.matmul(l[j - 1], w[j]) + b[j])
                else:
                    l[j] = tf.nn.tanh(tf.matmul(l[j - 1], w[j]) + b[j])
            if j == len(w) - 1:
                if j in net.get_relu():
                    if final:
                        print('goes relu!!!')
                    predict = tf.nn.relu(tf.matmul(l[j - 1], w[j]) + b[j])
                elif j in net.get_elu():
                    predict = tf.nn.elu(tf.matmul(l[j - 1], w[j]) + b[j])
                else:
                    predict = tf.nn.tanh(tf.matmul(l[j - 1], w[j]) + b[j])
        error = tf.losses.mean_squared_error(predict, output)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(error)
        init_error = tf.constant(sess.run(error, feed_dict={input: test_feature, output: test_label}))
        #print(sess.run(error, feed_dict={input: test_feature, output: test_label}))
        #print(sess.run(w))
        for j in range(step):
            sess.run(optimizer, feed_dict={input: train_feature, output: train_label})
        net_error = sess.run(error, feed_dict={input: test_feature, output: test_label})
        error_diff = sess.run(init_error - error, feed_dict={input: test_feature, output: test_label})
        if(final):
            return sess.run(w), sess.run(b), net_error, error_diff, sess.run(predict, feed_dict={input: test_feature, output: test_label})*50, test_label*50
        return sess.run(w), sess.run(b), net_error, error_diff

    def net_updated(self, net, w, b, net_error=0, error_diff=0, init=False):
        try:
            net.set_w(w, init)
            net.set_b(b, init)
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
            #print('kjkjlkjk', size)
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

    def node_deletion(self, net, is_random=True, to_detele_num=0, layer=2):
        w, b = net.get_all()
        if net.get_nodesnum(layer=layer) < 2:
            print('only one node, cannot be deleted')
            print(w)
        if is_random and net.get_nodesnum() >= 2:
            to_detele_num = random.randint(1, net.get_nodesnum(layer=layer) - 1)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        new_w = []
        new_b = []
        for i in range(len(sess.run(w))):
            if layer - 2 == i:
                #print(np.shape(sess.run(w)[i]), np.shape(sess.run(w)[i][:, 0:-to_detele_num]))
                if net.get_nodesnum() >= 2:
                    new_w.append(sess.run(w)[i][:, 0:-to_detele_num])
                    new_b.append(sess.run(b)[i][0:-to_detele_num])
                else:
                    new_w.append(sess.run(w)[i][:])
                    new_b.append(sess.run(b)[i][:])
            elif layer - 1 == i:
                if net.get_nodesnum() >= 2:
                    new_w.append(sess.run(w)[i][0:-to_detele_num])
                    new_b.append(sess.run(b)[i])
                else:
                    new_w.append(sess.run(w)[i][:])
                    new_b.append(sess.run(b)[i])
            else:
                new_w.append(sess.run(w)[i])
                new_b.append(sess.run(b)[i])
        #print('deletion done')
        return new_w, new_b

    def node_addition(self, net, is_random=True, to_add_num=0, layer=3):
        w, b = net.get_all()
        if is_random:
            to_add_num = random.randint(1, 3)
        new_w = []
        new_b = []
        #print(w)
        #print(net.get_nodesnum(layer-1))
        initial = tf.random_uniform((net.get_nodesnum(layer=layer-1), to_add_num))
        add_w1 = tf.Variable(initial)
        initial = tf.random_uniform((to_add_num, net.get_nodesnum(layer+1)))
        add_w2 = tf.Variable(initial)
        initial = tf.constant(0.1, shape=(to_add_num, ))
        add_b = tf.Variable(initial)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for i in range(len(sess.run(w))):
            if layer - 2 == i:
                #print(np.shape(sess.run(w)[i]), np.shape(sess.run(add_w1)))
                new_w.append(np.concatenate((sess.run(w)[i], sess.run(add_w1)), axis=1))
                new_b.append(np.concatenate((sess.run(b)[i], sess.run(add_b)), axis=0))
            elif layer - 1 == i:
                new_w.append(np.concatenate((sess.run(w)[i], sess.run(add_w2)), axis=0))
                new_b.append(sess.run(b)[i])
            else:
                new_w.append(sess.run(w)[i])
                new_b.append(sess.run(b)[i])
        return new_w, new_b


def probabality_generator(max=3):
    total = reduce(lambda x, y: x + y, range(max+1))
    get_prob = random.randint(1, total)
    for p in range(1, max+1):
        if get_prob > sum(range(p)) and get_prob <= sum(range(p + 1)):
             return max - p

def get_rid_of_same(populations):
    print(len(populations))
    for i in range(len(populations)):
        for j in range(i+1, len(populations)):
            if(populations[i].get_rank() == populations[j].get_rank()):
                print("delete", i)
                populations.pop(i)
                i -= 1
                print(len(populations))
    return populations


def evolve(population_nums, generations, tra_feature, tra_label, vaildation_feature, vaildation_label, test_feature, test_label, type='optimal', ):

    populations = Population(population_nums, 6, 1).nets
    populations = Evolution(population_nums).population_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, populations, 50)
    populations = Evolution(population_nums).state_marker(populations)
    populations = Evolution(population_nums).rank(populations)
    populations = get_rid_of_same(populations)
    populations = Evolution(population_nums).sort_nets(populations)




    for generation in range(generations):
        if generation%10 == 0:
            for i in range(len(populations)):
                print('ididididididididid', populations[i].get_id())
        if type=='random':
            i = probabality_generator(max=population_nums)
        if type=='optimal':
            i = 0
        populations = Evolution(population_nums).rank(populations)
        populations = Evolution(population_nums).sort_nets(populations)

        replacement = 0


        print('error of network ', i, 'in generation ', generation + 1, 'is ', populations[i].get_error())
        w, b, e, r = Evolution().network_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, populations[i], 30)
        if r > 0.0005:
            #print(r)
            print('update the weight for network ', i, 'in generation', generation+1)
            Evolution().net_updated(populations[i], w, b, e, r)
            print('\n')



        else:
            for layer in range(2, populations[i].get_layer() + 1):
                act_change = 0
                if layer-2 in populations[i].get_elu():
                    act_change = 1
                    populations[i].set_elu(layer - 2, op='remove')
                if layer-2 in populations[i].get_relu():
                    populations[i].set_relu(layer - 2, op='remove')
                    w, b, e, r = Evolution().network_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, populations[i], 40)
                    if r > 0.0005:
                        # print(r)
                        print('act function changes to tanh network ', i,'at layer', layer, 'in generation', generation+1)
                        Evolution().net_updated(populations[i], w, b, e, r)
                        print('\n')
                        replacement = 1
                        break
                populations[i].set_relu(layer-2)
                w, b, e, r = Evolution().network_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, populations[i], 40)
                if r > 0.0005:
                    print('act function changes to relu network ', i,'at layer', layer, 'in generation', generation+1)
                    Evolution().net_updated(populations[i], w, b, e, r)
                    print('\n')
                    replacement = 1
                    break
                else:
                    if act_change == 1:
                        populations[i].set_elu(layer - 2)
                    populations[i].set_relu(layer-2, op='remove')

            if replacement == 1:
                print('\n')
                continue

            for layer in range(2, populations[i].get_layer() + 1):
                act_change = 0
                if layer-2 in populations[i].get_relu():
                    act_change = 1
                    populations[i].set_relu(layer - 2, op='remove')
                if layer-2 in populations[i].get_elu():
                    populations[i].set_elu(layer - 2, op='remove')
                    w, b, e, r = Evolution().network_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, populations[i], 50)
                    if r > 0.0005:
                        # print(r)
                        print('act function changes to tanh network ', i,'at layer', layer, 'in generation', generation+1)
                        Evolution().net_updated(populations[i], w, b, e, r)
                        print('\n')
                        replacement = 1
                        break

                populations[i].set_elu(layer-2)
                #print(populations[i].get_elu())
                w, b, e, r = Evolution().network_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, populations[i], 50)
                if r > 0.0005:
                    print('act function changes to elu network ', i,'at layer', layer, 'in generation', generation+1)
                    Evolution().net_updated(populations[i], w, b, e, r)
                    print('\n')
                    replacement = 1
                    break
                else:
                    if act_change == 1:
                        populations[i].set_relu(layer - 2)
                    populations[i].set_elu(layer-2, op='remove')

            if replacement == 1:
                print('\n')
                continue

            # TODO: SA training and Replacement
            for layer in range(2, populations[i].get_layer() + 1):
                #print(layer, populations[i].get_layer())
                w, b = Evolution().node_deletion(populations[i], layer=layer)      # delete nodes
                temp_net = Population(1, 6, 1, layer_num=populations[i].get_layer()+1, random_layer=False).nets[0]
                Evolution().net_updated(temp_net, w, b, init=True)
                temp_net.set_relu(populations[i].get_relu(), copy=True)
                temp_net.set_elu(populations[i].get_elu(), copy=True)
                w, b, e, r = Evolution().network_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, temp_net, random.randint(50, 60) * generation)
                if e < populations[-1].get_error() and not check_same(temp_net, populations):
                    print('node deleted for network ', i,'at layer', layer, 'in generation', generation+1)
                    #Evolution().net_updated(populations[-1], w, b, e, r)
                    Evolution().net_updated(temp_net, w, b, e, r)
                    temp_net.set_id(populations[i].get_id())
                    populations.pop()
                    populations.append(temp_net)
                    replacement = 1
                    break

            if replacement == 1:
                print('\n')
                continue

            for layer in range(2, populations[i].get_layer() + 1):
                # TODO: connection deletion
                w, b = Evolution().node_addition(populations[i], layer=layer)    # add nodes
                temp_net = Population(1, 6, 1, layer_num=populations[i].get_layer() + 1, random_layer=False).nets[0]
                Evolution().net_updated(temp_net, w, b, init=True)
                temp_net.set_relu(populations[i].get_relu(), copy=True)
                temp_net.set_elu(populations[i].get_elu(), copy=True)
                w, b, e, r = Evolution().network_optimizer(tra_feature, tra_label, vaildation_feature, vaildation_label, temp_net, random.randint(50, 60) * generation)
                if e < populations[-1].get_error() and not check_same(temp_net, populations):
                    print('node added for network ', i, 'at layer', layer, 'in generation', generation + 1)
                    #Evolution().net_updated(populations[-1], w, b, e, r)
                    Evolution().net_updated(temp_net, w, b, e, r)
                    temp_net.set_id(populations[i].get_id())
                    populations.pop()
                    populations.append(temp_net)
                    replacement = 1
                    break
            if replacement == 0:
                print('no changed')



            print('\n')

    populations = Evolution(population_nums).rank(populations)
    populations = Evolution(population_nums).sort_nets(populations)
    return populations[0]



def check_same(net, populations):
    for err in range(len(populations)):
        if net.get_error() == populations[err].get_error():
            return True
    return False









