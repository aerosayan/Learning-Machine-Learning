# FILE : 01_soup_sale_gradient_descent.py
# AUTH : Sayan Bhattacharjee
# EMAIL: aero.sayan@gmail.com
# DATE : 2/JULY/2018 MON
# INFO : How does hot soup sale change in winter based on temperature?
#      : Here, we do linear regression with gradient descent

import numpy as np
import matplotlib.pyplot as plt

def compute_error(_m,_b,_x,_y):
    # Sum of squared errors
    sum_err = 0
    n = len(_x)
    if(n == len(_y)):
        for i in range(n):
            sum_err += (_y[i]-(_m*_x[i] + _b))**2.0
    else:
        print("ERR : length mismatch...")
        aseert(False)
    return sum_err / float(n)

def run_gradient_descent(_x,_y,_m,_b,_rate,_num_it):
    # Do gradient descent
    m = 0
    b = 0
    for i in range(_num_it):
        m,b = gradient_descent_step(m,b,_x,_y,_rate)
        if(i%(_num_it / 8)==0):
            plt.plot(_x,m*_x+b,'-b',alpha=i/float(_num_it))

    return (m,b)

def gradient_descent_step(_m,_b,_x,_y,_rate):
    # Do one gradient descent step
    m_grad = 0
    b_grad = 0

    # To calculate the gradients we need to calculate the partial
    # # derivatives of m and b
    n = len(_x)
    sum_m = 0
    sum_b = 0

    for i in range(n):
        guess = (_m*_x[i] + _b)
        error = _y[i] - guess
        sum_m += -1 * (error)*_x[i]
        sum_b += -1 * (error)
    m_grad = (2/float(n))*sum_m
    b_grad = (2/float(n))*sum_b

    m_new = _m - (_rate*m_grad)
    b_new = _b - (_rate*b_grad)

    return(m_new,b_new)


if __name__ == "__main__":
    n = 100                                               # no. of data points
    temp = np.linspace(3, 30, n)                          # temperature (deg C)
    noise = np.random.randint(-5,7,size = n)              # noise to simulate RL
    soup = np.linspace(40, 22 , n, dtype = 'int') + noise # soup sale count

    # We are re-assigning the data since we like to write in this form
    x = temp                                                 # x co-ordinate
    y = soup                                                 # y co-ordinate

    init_m = 0                                               # slope guess
    init_b= 0                                                # intercept guess
    init_err = compute_error(init_m,init_b,x,y)              # initial error
    rate = 0.0005                                            # learning rate

    msg1 = "Starting gradient descent with m : {m_val},b : {b_val}"+ \
    " and error :{err_val}"

    print(msg1.format(m_val = init_m,b_val = init_b, err_val = init_err))

    # number of iterations
    num_it = 1000*15
    [m,b]  = run_gradient_descent(x,y,init_m,init_b,rate,num_it)
    msg2 = "Ran {n} iterations of gradient descent and got m : {m_val}"+\
    ",b : {b_val} and error :{err_val}"

    final_err = compute_error(m,b,x,y)                       # final error
    print(msg2.format(n=num_it,m_val = m,b_val = b,err_val = final_err))

    # Plot the results
    plt.grid()
    plt.scatter(x,y)
    plt.plot(x,m*x + b,'-r',linewidth = 3.0)
    plt.title("Temperature vs Soup Sale"+ \
    " \n (linear regression with gradient descent)"+\
    " \n (Iteration : "+str(num_it)+") (Learning Rate : "+str(rate)+")")
    plt.xlabel("Temp in degree Celcius")
    plt.ylabel("Hot soup bowls sold")
    print("The linear regression has resulted in ...")
    print("m(slope)       : ",m )
    print("b(y intercept) : ",b )


    plt.show()
