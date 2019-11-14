import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def one_hot_encode(y):
    n_labels = len(y)
    n_unique_labels = len(np.unique(y))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), y] = 1
    return one_hot_encode


# Dataset Read
def read_data():
    df = pd.read_csv(".\\TrainingModels\\points-2.csv")
    X = df[df.columns[0:2]].values
    y = df[df.columns[2]]
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    return(X,Y)

X, Y = read_data()

X, Y = shuffle(X, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)


l = 0.9
epochs = 1500
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
    hidden_layer1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    hidden_layer1 = tf.nn.sigmoid(hidden_layer1)

    hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['h2']), biases['b2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)

    output_layer = tf.matmul(hidden_layer2, weights['out'] + biases['out'])

    return output_layer



weights = {

    'h1' : tf.Variable(tf.truncated_normal([n_dim, n_hidden1])),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2])),
    'out' : tf.Variable(tf.truncated_normal([n_hidden2, nb_class]))
}


biases = {

    'b1' : tf.Variable(tf.truncated_normal([n_hidden1])),
    'b2' : tf.Variable(tf.truncated_normal([n_hidden2])),
    'out' : tf.Variable(tf.truncated_normal([nb_class]))
}


init = tf.global_variables_initializer()

save = tf.train.Saver()

y = perceptron(x, weights, biases)



cost_calc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(l).minimize(cost_calc)

sess = tf.Session()
sess.run(init)


mse_history = []
accuracy_history = []



for epoch in range(epochs):
    sess.run(training_step, feed_dict={x: train_x, y_:train_y})
    cost = sess.run(cost_calc, feed_dict={x: train_x, y_:train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)

    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)

    print('iteration : ', epoch, '  -  ' ,  "- Accuracy: ", accuracy)


save_path = save.save(sess, model_path)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("\n\n\n""Final accuracy : ", sess.run(accuracy, feed_dict={x: test_x, y_:test_y}))

