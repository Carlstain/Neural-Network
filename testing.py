import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#Testing line !

def one_hot_encode(y):
    n_labels = len(y)
    n_unique_labels = len(np.unique(y))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), y] = 1
    return one_hot_encode


# Dataset Read
def read_data():
    df = pd.read_csv(".\\TestModels\\points-3.csv")
    X = df[df.columns[0:2]].values
    y1 = df[df.columns[2]]
    # print(y)
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    print(X.shape)

    return (X, Y, y1)


X, Y, y1 = read_data()

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=415)

l = 0.9
epochs = 100
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
nb_class = 4
model_path = ".\\Model"

n_hidden1 = 2
n_hidden2 = 2

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, nb_class]))
b = tf.Variable(tf.zeros([nb_class]))
y_ = tf.placeholder(tf.float32, [None, nb_class])


def perceptron(x, weights, biases):
    hidden_layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer1 = tf.nn.sigmoid(hidden_layer1)

    hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['h2']), biases['b2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)

    output_layer = tf.matmul(hidden_layer2, weights['out'] + biases['out'])

    return output_layer


weights = {

    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.truncated_normal([n_hidden2, nb_class]))
}

biases = {

    'b1': tf.Variable(tf.truncated_normal([n_hidden1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden2])),
    'out': tf.Variable(tf.truncated_normal([nb_class]))
}

init = tf.global_variables_initializer()
save = tf.train.Saver()
y = perceptron(x, weights, biases)

cost_calc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(l).minimize(cost_calc)

sess = tf.Session()
sess.run(init)
save.restore(sess, model_path)

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(0, len(X)):

    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 2)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 2), y_: Y[i].reshape(1, 4)})

    if prediction_run == 0 and accuracy_run == 1:
        plt.plot([X[i][0]], [X[i][1]], 'b+')

    elif prediction_run == 1 and accuracy_run == 1:
        plt.plot([X[i][0]], [X[i][1]], 'y+')

    elif prediction_run == 2 and accuracy_run == 1:
        plt.plot([X[i][0]], [X[i][1]], 'r*')

    elif prediction_run == 3 and accuracy_run == 1:
        plt.plot([X[i][0]], [X[i][1]], 'g*')

    else:
        plt.plot([X[i][0]], [X[i][1]], 'ko')

    print("Position : ", X[i], "\tOriginal Class : ", y1[i], "\tPredicted Values : ", prediction_run, "  Accuracy: ",
          accuracy_run, "\n")

"""
    if prediction_run == 1 and accuracy_run == 1:
        plt.plot([X[i][0]],[X[i][1]],'r*')

    elif prediction_run == 0 and accuracy_run == 1:
        plt.plot([X[i][0]], [X[i][1]], 'b+')

    else:
        plt.plot([X[i][0]], [X[i][1]], 'ko')

"""

plt.show()
