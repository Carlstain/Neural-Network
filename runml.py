from testing import Testing
from training import Training

iterations = 1200
nb_class = 6
n_hidden1 = 15
n_hidden2 = 15
learningrate = 0.9

training = Training(iterations, learningrate, nb_class, n_hidden1, n_hidden2)
testing = Testing(learningrate, nb_class, n_hidden1, n_hidden2)

training.train()
testing.prediction()
