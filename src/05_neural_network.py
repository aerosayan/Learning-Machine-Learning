# LANG : Python 2.7
# FILE : 05_neural_network.py
# AUTH : Sayan Bhattacharjee
# EMAIL: aero.sayan@gmail.com
# DATE : 6/JULY/2018             (Started Creation  )
# DATE : 11/JULY/2018            (Finished Creation )
# DATE : 11/JULY/2018            (Last Modified     )
# INFO : Neural network - A multi-level fully connected neural network.
#      : A multi-level neural network with feed forward architechture and
#      : backpropagation learning system.
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

class Layer:
    """ Neural network layer
    @param _n : number of neurons to create in that layer
    """
    def __init__(self,_nl,_nlminus1):
        """ Constructor """
        # -----------------------------------------------------
        # Important parameters and outputs
        # -----------------------------------------------------
        # No. of neurons in the current layer(L)
        self.nl = _nl
        # No. of neurons in the  previous layer(L-1)
        self.nlminus1 = _nlminus1
        # The weight numpy matrix for the layer
        self.w = []
        # The bias numpy column matrix
        self.b = []
        # The weighted sum vector
        self.z = []
        # The activation vector of the layer
        self.a = []

        # -----------------------------------------------------
        # Gradients of the cost with respect to bias and weight
        # -----------------------------------------------------
        # The error thingy: I forgot the original name. DOH!! Stupid Homer! LOL
        self.delta = []
        # The gradient of cost with respect to bias
        self.grad_b = []  # NOTE : grad_b is actually equal to self.delta
        # The gradient of cost with respect to weight
        self.grad_w = []

    def init_matrices(self):
        """ Initialize the weight and bias matrices """
        self.init_w()    # Initialize the weight matrix
        self.init_b()    # Initialize the bias vector
    def init_w(self):
        """ Initialize weight matrix """
        nl = self.nl
        nlminus1 = self.nlminus1
        self.w = np.matrix(np.random.uniform(low=-1,\
        high=+1,size=nl*nlminus1).reshape(nl,nlminus1))

    def init_b(self):
        """ Initialize bias vector """
        nl = self.nl
        nlminus1 = self.nlminus1
        self.b = np.matrix(np.random.uniform(low=-1,\
        high=+1,size=nl).reshape(nl,1))

    def weighted_sum(self,_a_lminus):
        """ Calculate and return the weighted sum of the function
        @param _a_lminus : the activation of the previous layer
        """
        # NOTE : all parameters should be numpy matrix
        return np.matrix(np.dot(self.w,_a_lminus) + self.b)

    def relu(self,_z):
        """ Calculate ReLU activation of the layer based on the weighed sum
        @param _z : the weighed sum of the current layer
        """
        # Create and assign space for activation
        a = _z.copy()
        # Apply vectorized form of ReLU = max(0,z)
        a[_z <= 0 ] = 0
        return np.matrix(a)

    def relu_p(self,_z):
        """ Create ReLU derivative of the layer based on the weighed sum
        @param _z : the weighed sum of the current layer
        """
        rprime = _z.copy()
        # Apply the derivative of ReLU
        rprime[rprime <= 0 ] = 0
        rprime[rprime >  0 ] = 1
        return np.matrix(rprime)
    def calc_delta_last(self,_y):
        """ Calculate delta for the last layer in the neural network
        @ param : _y : Labeled desired data
        """
        self.delta = np.matrix( np.multiply\
        (self.a - _y,self.relu_p(self.z)) )

    def calc_delta(self,_del_lplus):
        """ Calculate general purpose delta for the internal layers.
        @ param : _del_lplus : the delta vector of the next layer
        """
        self.delta = np.matrix(np.multiply( \
        np.dot(self.w.T,_del_lplus), self.relu_p(self.z)) )

    def calc_bias_grad(self):
        """ Calculate the gradient(derivative) of cost wrt bias """
        self.grad_b = np.matrix(self.delta.copy() )

    def calc_weight_grad(self,_a_lminus):
        """ Calculate the gradient(derivative) of cost wrt weight
        @param _a_lminus : the activation of the layer L-1
        """
        self.grad_w = np.matrix(np.dot(self.delta,_a_lminus.T ) )

class NeuralNetwork:
    """ Multi layer neural network """
    def __init__(self,_x,_y,_nhnodes,_nhlayers = 1):
        """ Constructor for the neural network class
        @param _x : input in np.matrix row vector format
        @param _y : output target in np.matrix row vector format
        @param _nhnodes : number of nodes in each hidden layer
        @param _nhlayers : number of hidden layers(default = 2)
        """
        # -----------------------------------------------------
        # Data sanity check
        # -----------------------------------------------------
        if(_x.shape[1] != _y.shape[1] ):
            print("ERR : The number of inputs and target labels mismatch")
            print("ERR : Cannot perform supervised learing ")
            print("ERR : Fatal error ! Exiting program ...")
            assert(False)

        # -----------------------------------------------------
        # Important parameters and outputs
        # -----------------------------------------------------
        # Input data stored in column vector form
        # x = |x11 x21   <- (COLUMNS ) Multiple study cases
        #     |x12 x22
        #     |x13 x23
        #         ^_(COLUMN VECTORS) data points in each study
        self.x = np.matrix(_x)

        # Target data stored in column vector form
        # y = |y11 y21   <- (COLUMNS) Multiple study cases
        #     |y12 y22
        #     |y13 y23
        #         ^_(COLUMN VECTORS) data points in each study
        self.y = np.matrix(_y)
        # The number of test cases and data points
        self.n_cases = self.x.shape[1]
        # The number nodes in input
        self.n_inodes  = self.x.shape[0]
        # The number of nodes hidden in each hidden layer
        self.n_hnodes  = _nhnodes
        # The number of nodes in output
        self.n_onodes  = self.y.shape[0]
        # The number of hidden layers
        self.n_hlayers = _nhlayers

        # -----------------------------------------------------
        # Layers
        # -----------------------------------------------------
        # The objects of the layer class
        self.layers = np.array([])

    def print_all(self,_verbose=False):
        """ Print all the data members based on the verbosity input"""
        if _verbose : print("x : \n",self.x)
        if _verbose : print("y : \n",self.y)
        print("number of input nodes  : ",self.n_inodes)
        print("number of hidden nodes : ",self.n_hnodes)
        print("number of output nodes : ",self.n_onodes)
        print("number of hidden  layers : ",self.n_hlayers)

    def create_layers(self):
        """ Create all the necessary layers """
        for i in range(self.n_hlayers):
            if i == 0:
                # Add first hidden layer
                print("INF : Creating first hidden layer...")
                self.layers = np.append(self.layers,Layer(self.n_hnodes,self.n_inodes))
                self.layers[-1].init_matrices()
                print(self.layers[-1].w)
                continue
            else:
                # Add all else hidden layers
                print("INF : Creating internal hidden layer...")
                self.layers = np.append(self.layers,Layer(self.n_hnodes,self.layers[-1].w.shape[0]))
                self.layers[-1].init_matrices()
                print(self.layers[-1].w)
                continue
        # Add last output layer
        print("INF : Creating output layer...")
        self.layers = np.append(self.layers,Layer(self.n_onodes,self.layers[-1].w.shape[0]))
        self.layers[-1].init_matrices()
        print(self.layers[-1].w)
        print(self.layers)

    def train(self,_epochs = 1):
        """ Train the neural network """
        for epoch in range(_epochs):  # iterate through epochs
            for case in range(self.x.shape[1]): # iterate through all the cases
                print("DBG : Performing one forward propagation loop...")
                self.forward_propagation(case)
                print("DBG : Performing one back propagation loop...")
                self.back_propagation(case)

    def forward_propagation(self,_case,_verbose = False):
        """ Perform neural network forward propagation operation"""
        if _verbose : print("----------------------------")
        a = self.x[:,_case] # input layer values
        for j in xrange(len(self.layers)): # iterate through the full neural net
            if(j >= 1):
                a = self.layers[j-1].a # activation of hidden layers
            if _verbose :
                print("DBG : Before weighted sum...")
                print("DBG : z : ",self.layers[j].z)
                print("DBG : a : ",self.layers[j].a)

            # Calculate weighted sum and calculate activation function
            layer = self.layers[j]
            layer.z = layer.weighted_sum(a)
            layer.a = layer.relu(layer.z)

            if _verbose :
                print("DBG : After weighted sum...")
                print("DBG : z : ",self.layers[j].z)
                print("DBG : a : ",self.layers[j].a)
                print("----------------------------")


    def back_propagation(self,_case):
        """ Perform  neural network back propagation operation """
        learning_rate = 0.001
        y = self.y[:,_case]
        L = len(self.layers) -1
        for j in xrange(len(self.layers)-1,-1,-1): # Iterate back through layers
            layer = self.layers[j]
            if(j>0):
                a = self.layers[j-1].a    # activation of previous hidden layers


            # Calculate the delta values
            if(j == L): # if the last value
                layer.calc_delta_last(y) #ERROR  y size does not match a
            elif(j < L ): # All other layers before the last layer
                layer.calc_delta(self.layers[j+1].delta)

            # Calculate the necessary gradients
            if(j == 0 ): # If we are at the front of the hidden layer
                layer.calc_weight_grad(self.x[:,case])
            elif(j>0): # else if we are above j = 0
                layer.calc_weight_grad(a)

            layer.calc_bias_grad()

            # Update the weights
            layer.w += -layer.grad_w *learning_rate
            layer.b += -layer.grad_b *learning_rate





if __name__ == "__main__":
    x = np.matrix([[7,5,2],[6,7,8]])
    y = np.matrix([[99,60,35]])
    print(x.shape)
    print(y.shape)
    nn = NeuralNetwork(x,y,3,_nhlayers=4)
    nn.print_all(_verbose=True)
    nn.create_layers()
    nn.train()
