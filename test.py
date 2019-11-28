from testing import Testing

nb_class = 6
n_hidden1 = 30
n_hidden2 = 30
learningrate = 0.7

testing = Testing(learningrate, nb_class, n_hidden1, n_hidden2)
testing.prediction()
