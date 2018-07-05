# LANG : Python 2.7
# FILE : 04_perceptron_biased.py
# AUTH : Sayan Bhattacharjee
# EMAIL: aero.sayan@gmail.com
# DATE : 4/JULY/2018
# INFO : Perceptron- the simplest neural network with bias
#      : Here, we do binary classification using single layer perceptron
#      : But since I am amazing and you who is trying to learn a new thing
#      : are amazing, we will bias the activation of the perceptron to allow
#      : better activation control and convert all of the old methods to full
#      : numpy based linear algebra for super fast training speeds.
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

def linear(_x):
    """ Create and return in linear form of y = m*x + b """
    m = 2
    b = -0.1
    return m*_x + b

class Perceptron:
    def __init__(self):
        self.x = []                                  # Inputs
        self.w = []                                  # Weights
        self.y = []                                  # Output targets

    def print_all(self):
        print("DBG : self.x :")
        print(self.x)
        print("DBG : self.w :")
        print(self.w)
        print("DBG : self.y :")
        print(self.y)

    def calc_err(self):
        """ Calculate error in the perceptron
        """
        n = len(self.x)
        guess = self.guess()
        error = guess - self.y
        return np.sum((error)**2)/float(n)

    def activation(self,_activation_input):
        """ Activation function of the Perceptron """
        # If the guess and the bias added make the acitvation input go over
        # # or below zero then clasify them as either +1 or -1 respectively
        _activation_input[_activation_input >= 0 ] = +1
        _activation_input[_activation_input <  0 ] = -1
        return _activation_input

    def guess(self,_echo = False,_verbose = False):
        """ Guess and classify the inputs """
        # Calculate sum of the weighted inputs
        wsum = np.dot(self.x,self.w)

        if(_verbose): print("DBG : activation_input : wsum"); print(wsum);

        # Send to activation method and return guess
        activation_output = self.activation(wsum)
        if(_verbose): print("DBG : activation_output : guess :"); print(activation_output);
        return  activation_output

    def train(self,_learning_rate,_epoch = 1,_echo = False,_verbose = False):
        print("INF : Total epoch : "+ str(_epoch))

        # Create temp variable for weights
        weights = self.w

        # Train for a set number of epochs
        for i in range(_epoch):
            # Number of data points
            N = float(len(self.x))
            # Guess output and classify using the activation function
            guess = self.guess(_echo=_echo,_verbose=_verbose)
            # Calculate error from suprevised learning datasetknown results
            error =  self.y - guess

            # Update weights using gradient descent
            for k in range(len(weights)): # TODO : bias column handling
                weights[k] +=  2.0/N* (np.dot(self.x[:,k] , np.transpose(error)) * _learning_rate)
            self.w = weights
            # end for
        # end for



if __name__ == "__main__":
    #----------------------------------------------------------
    # Create training dataset
    #----------------------------------------------------------
    # Number of datasets
    n = 100*10
    # Range of x and y co-ordinates
    xmin = -1.0;xmax = 1.0;ymin = -1.0;ymax = 1.0;
    # Create x, y co-ordinates and labels for supervised learning
    x = np.random.uniform(low = xmin, high = xmax, size = n)
    y = np.random.uniform(low = ymin, high = ymax, size = n)
    labels = np.zeros(n)
    # Set the supervised training target data
    labels[y >= linear(x)] = +1
    labels[y <  linear(x)] = -1
    # Create random initial weights
    rand_weights = np.random.rand(3)   # [bias_weight,input_weight1,input_weight2]

    #----------------------------------------------------------
    # Create and train the perceptron
    #----------------------------------------------------------
    # Create perceprton object
    p = Perceptron()
    # Insert inputs,weights and labels for supervised learning
    p.x = np.column_stack((np.ones(n),x,y))   # [[bias] , [input0],...[inputN]]
    p.w = rand_weights
    p.y = labels
    p.b = 1

    print("INF : TRAINING INITIAL ERROR : %2.5f " % p.calc_err())
    p.train(_learning_rate = 0.001,_epoch = 5000,_echo = True)
    print("INF : TRAINING FINAL ERROR   : %2.5f " % p.calc_err())

    #----------------------------------------------------------
    # Validate the model with new data to prevent overfitting
    #----------------------------------------------------------
    # The size modifier
    m = 2
    # the x and  y co-ordinates
    x = np.random.uniform(low=xmin,high=xmax,size=(n/m))
    y = np.random.uniform(low=ymin,high=ymax,size=(n/m))

    # insert  new test data
    p.x = np.column_stack((np.ones(n/m),x,y))   # stack the inputs as columns
    # NOTE : WE DO NOT CHANGE THE TRAINED WEIGHTS CAUSE THAT IS WHAT WE WANT

    # Guess a classification on the new data set
    guess = p.guess()

    # Plot the final results
    plt.figure()
    plt.title("Biased Single Layer Perceptron Classifier")
    red_pos_x= x[guess == +1]
    red_pos_y= y[guess == +1]
    black_pos_x= x[guess == -1]
    black_pos_y= y[guess == -1]
    plt.scatter(red_pos_x,red_pos_y,color='r')
    plt.scatter(black_pos_x,black_pos_y,color='k')


    # Plot the dividing line
    plt.plot([xmin,xmax],[linear(xmin),linear(xmax)],'b-',linewidth=7)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.show()
