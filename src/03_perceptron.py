# LANG : Python 2.7
# FILE : 03_perceptron.py
# AUTH : Sayan Bhattacharjee
# EMAIL: aero.sayan@gmail.com
# DATE : 3/JULY/2018 
# INFO : Perceptron- the most simplest neural network. $#*& just  got serious.
#      : Here, we do classification with gradient descent
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.target = 0

    def guess(self):
        """ Guess the result using the weights and classify as either +1 or -1
        """
        n = len(self.weights)
        _weights = self.weights
        wsum = 0
        for i in range(n):
            # calculate weighted sum
            wsum += self.inputs[i] * _weights[i]
            # activation function is a simple sign check

        output = self.sign(wsum)
        return output

    def train(self):
        """ Train the perceptron
        """
        _guess = self.guess()
        error  = self.target - _guess
        n = len(self.weights)
        _weights = self.weights
        learn_rate = 0.001
        for i in range(n):
            _weights[i] += error * self.inputs[i]*learn_rate

        self.weights = _weights


    def calc_error(self,_x_vec,_y_vec,_targets):
        """ Calculate error in the perceptron
        """
        n = len(_x_vec)
        errsum = 0
        for i in  range(n):
            self.inputs = np.array([_x_vec[i],_y_vec[i]])
            _guess = self.guess()
            error  = _targets[i] - _guess
            errsum += error**2
        return errsum/float(n)

    def sign(self,_number):
        """ return 1 if sign is positive and return -1 if sign is negative
        """
        if(_number >=0 ):
            return +1
        else:
            return -1




if __name__ == "__main__":
    # Generate Perceptron object
    p = Perceptron()

    # Create random weights
    weights = np.random.rand(2)              # will convserge to [-0.144,0.144]
    # Assign weights to perceptron
    p.weights = weights

    ## Create the labelled data set for supervised learning
    # no. of data points to be created and used
    n = 100*20
    print("creating perceptron for :",n,"nodes")
    # the x co-ordinates
    x = np.random.uniform(low=0,high=1,size=(n))
    # the y co-ordinates
    y = np.random.uniform(low=0,high=1,size=(n))
    # labesl for supervised learning for the perceptron
    labels = np.zeros(n)
    # assign valus of +1 or -1 to the labels
    labels[y >x] = +1
    labels[y <x] = -1

    N = len(x)
    print("initial error :",p.calc_error(x,y,labels))
    print("initial weights :",p.weights)
    for i in range(N):
        # Call the training
        p.inputs = np.array([x[i],y[i]])          # set the input
        p.target = labels[i]                      # set the label target
        p.train()                                 # train the weights

    print("final weights :",p.weights)
    print("final error :",p.calc_error(x,y,labels))

    # the x co-ordinates
    x = np.random.uniform(low=0,high=1,size=(n))
    # the y co-ordinates
    y = np.random.uniform(low=0,high=1,size=(n))

    # Plot the final results
    plt.scatter(0,0)
    for i in range(n):
        # Call the training
        p.inputs = np.array([x[i],y[i]])          # set the input
        guess = p.guess()
        if(guess == 1):
            # if of class 1 then plot in red
            plt.scatter(x[i],y[i],color="r")
            continue
        elif(guess == -1):
            # if of class -1 plot in black
            plt.scatter(x[i],y[i],color="k")
            continue



    # Plot the remaining stuff
    plt.plot([0,1],[0,1],'b-',linewidth=7)
    plt.title("Perceptron based classification")
    plt.show()
