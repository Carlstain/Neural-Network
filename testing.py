import pandas as pd
import matplotlib.pyplot as plt
from utilities import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Dataset Read
def read_data():
    df1 = pd.read_csv("./TestModels/X_test.csv")
    df2 = pd.read_csv("./TestModels/y_test.csv")
    X = df1.values
    y1 = df2.values
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)

    return (X, Y, y1)


class Testing:

    def __init__(self, learningrate, nb_class, n_hidden1, n_hidden2, model_path="./Model/Model"):
        self.learningrate = learningrate
        self.nb_class = nb_class
        self.model_path = model_path
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2

    def prediction(self):
        X, Y, y1 = read_data()

        train_test_split(X, Y, test_size=0.3, random_state=415)

        n_dim = X.shape[1]
        x = tf.placeholder(tf.float32, [None, n_dim])
        tf.Variable(tf.zeros([n_dim, self.nb_class]))
        tf.Variable(tf.zeros([self.nb_class]))
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
        tf.train.GradientDescentOptimizer(self.learningrate).minimize(cost_calc)

        self.sess = tf.Session()
        self.sess.run(init)
        save.restore(self.sess, self.model_path)

        prediction = tf.argmax(y, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        classpredictiondistribution = {}
        for classe in range(1, 7):
            classpredictiondistribution[classe] = {'correct': 0, 'total': 0}
        for i in range(0, len(X)):
            print_progress(i, len(X)-1, prefix="Testing", suffix="Complete")
            accuracy_run = self.sess.run(accuracy, feed_dict={x: X[i].reshape(1, 561), y_: Y[i].reshape(1, 6)})
            classpredictiondistribution[y1[i][-1]]['total'] += 1
            if bool(accuracy_run):
                classpredictiondistribution[y1[i][-1]]['correct'] += 1
        distribution = []
        total = 0
        for key in classpredictiondistribution:
            total += classpredictiondistribution[key]['correct']
            distribution.append(classpredictiondistribution[key]['correct']/classpredictiondistribution[key]['total'])
        plt.plot([x for x in range(1, 7)], distribution)
        plt.xlabel('Classes')
        plt.ylabel('Prediction accuracy')
        plt.savefig('results/Testing.png')
        print("Overall Prediction Accuracy (%) : ", total/len(X))

