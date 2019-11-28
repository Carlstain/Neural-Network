import pandas as pd
import matplotlib.pyplot as plt
from utilities import *
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Dataset Read


class Training:

    def __init__(self, iterations, learningrate, nb_class, n_hidden1, n_hidden2, model_path="./Model/Model"):
        self.model_path = model_path
        self.accuracy_history = []
        self.classdistribution = []
        self.iterations = iterations
        self.nb_class = nb_class
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.learningrate = learningrate

    def drawplot(self):

        plt.subplot(2, 1, 1)
        plt.plot(self.accuracy_history, 'r-')
        plt.xlabel('Iterations')
        plt.ylabel('Precision')
        plt.subplot(2, 1, 2)
        plt.plot([x for x in range(1, 7)], self.classdistribution, 'o-')
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.savefig('results/Training.png')

    def read_data(self):
        df1 = pd.read_csv(".\\TrainingModels\\X_train.csv")
        df2 = pd.read_csv(".\\TrainingModels\\y_train.csv")
        X = df1.values
        y = df2.values
        for value in range(1, 7):
            self.classdistribution.append(list(df2.values).count(value)/len(df2.values))

        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        Y = one_hot_encode(y)

        return X, Y

    def train(self):
        X, Y = self.read_data()
        X, Y = shuffle(X, Y, random_state=1)
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)

        n_dim = X.shape[1]

        x = tf.placeholder(tf.float32, [None, n_dim])
        W = tf.Variable(tf.zeros([n_dim, self.nb_class]))
        b = tf.Variable(tf.zeros([self.nb_class]))
        y_ = tf.placeholder(tf.float32, [None, self.nb_class])

        weights = {

            'h1': tf.Variable(tf.truncated_normal([n_dim, self.n_hidden1])),
            'h2': tf.Variable(tf.truncated_normal([self.n_hidden1, self.n_hidden2])),
            'out': tf.Variable(tf.truncated_normal([self.n_hidden2, self.nb_class]))
        }

        biases = {

            'b1': tf.Variable(tf.truncated_normal([self.n_hidden1])),
            'b2': tf.Variable(tf.truncated_normal([self.n_hidden2])),
            'out': tf.Variable(tf.truncated_normal([self.nb_class]))
        }

        init = tf.global_variables_initializer()

        save = tf.train.Saver()

        y = perceptron(x, weights, biases)

        cost_calc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        training_step = tf.train.GradientDescentOptimizer(self.learningrate).minimize(cost_calc)

        self.sess = tf.Session()
        self.sess.run(init)

        print('please wait...')

        for iteration in range(self.iterations):
            print_progress(iteration, self.iterations-1, prefix='Learning', suffix='Complete')
            self.sess.run(training_step, feed_dict={x: train_x, y_: train_y})
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy = (self.sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
            self.accuracy_history.append(accuracy)

        save.save(self.sess, self.model_path)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("\n\n\n""Final accuracy : ", self.sess.run(accuracy, feed_dict={x: test_x, y_:test_y}))
