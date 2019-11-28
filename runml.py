from os import system, chdir, listdir, startfile
from training import Training
from time import time

iterations = 1200
nb_class = 6
n_hidden1 = 30
n_hidden2 = 30
learningrate = 0.7

start = time()
training = Training(iterations, learningrate, nb_class, n_hidden1, n_hidden2)
training.train()
trainingtime = time() - start
training.drawplot()
start = time()
system("python test.py")
testingtime = time() - start
chdir('./results')
print('Training Time(s) : %f --- Testing Time(s) %f' % (trainingtime, testingtime))
for image in listdir():
    startfile(image)
print('Results files have been opened.')