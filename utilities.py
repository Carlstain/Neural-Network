import tensorflow as tf
import numpy as np
import sys


def perceptron(x, weights, biases):
    hidden_layer1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    hidden_layer1 = tf.nn.sigmoid(hidden_layer1)

    hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['h2']), biases['b2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)

    output_layer = tf.matmul(hidden_layer2, weights['out'] + biases['out'])

    return output_layer


def one_hot_encode(y):
    n_labels = len(y)
    n_unique_labels = len(np.unique(y))
    ohe = np.zeros((n_labels, n_unique_labels))
    ohe[np.arange(n_labels), y] = 1
    return ohe


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '_' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s' % (prefix, bar, percents, '%'))
    if iteration == total:
        sys.stdout.write(' %s\n' % suffix)
    sys.stdout.flush()