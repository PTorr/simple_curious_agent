from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def squre_root(x):
    return np.sqrt(x)
def squre_root_to_derivative(x):
    return 0.5/np.sqrt(x)

def ReLU(x):
    x[x<0]=0
    x[x>=0] = x[x>=0]
    return x

def ReLU_to_derivative(x):
    xx = x.copy()
    xx[x < 0] = 0
    xx[x >= 0] = 1
    return xx

def initialize_synapses(network_architecture):
    n = len(network_architecture)-1
    synapses = [None] * n
    for i in range(n):
        synapses[i] = 2 * np.random.random((network_architecture[i], network_architecture[i+1])) - 1

    return synapses

def ann(x,y,synapses_):
    # Feed forward through layers 0, 1, and 2
    synapses = np.copy(synapses_)
    layers = [None] * (len(synapses) + 1)

    layers[0] = x.T
    for i, s in enumerate(synapses):
        layers[i+1] = ReLU(np.dot(layers[i], synapses[i]))

    error = np.abs(layers[-1] - y)

    return layers, error


def learner_ann(x, y, synapses, alpha, num_of_iterations, train = False):
    # randomly initialize our weights with mean 0
    synapses_ = np.copy(synapses[:])

    layers_delta = [None] * len(synapses)
    layers_error = [None] * len(synapses)
    for j in range(num_of_iterations):

        ### feed forward
        layers, ann_error = ann(x, y, synapses[:])
        layers_error[-1] = np.copy(ann_error)
        loss = np.sum(ann_error ** 2) / (len(y))
        if train:
            ### propagate errors backwards
            n = len(layers_error)-1
            for i in np.arange(n,-1,-1):
                if i != n:
                    layers_error[i] = layers_delta[i+1].dot(synapses[i+1].T)
                layers_delta[i] = layers_error[i] * ReLU_to_derivative(layers[i+1])

            ### update weights backwards
            for i in np.arange(len(synapses)-1,-1,-1):
                synapses[i] -= alpha * (layers[i].reshape(-1, 1).dot(layers_delta[i].reshape(1, -1)))

    return synapses_, layers, layers_delta, layers_error, loss


if __name__ == '__main__':
    pass
    # for first run
    # hl = {1: [50, 100, 140]}
    # hls = hl[1]
    # input_size = 2
    # synapse_0, synapse_1, synapse_2, synapse_3 = initialize_synapses(hls, input_size)
    #
    # for i in range(1000):
    #     l4_error, synapse_0, synapse_1, synapse_2, synapse_3 = learner(synapse_0, synapse_1, synapse_2, synapse_3)
    #     # print l4_error