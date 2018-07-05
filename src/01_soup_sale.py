# LANG : Python 2.7
# FILE : 01_soup_sale.py
# AUTH : Sayan Bhattacharjee
# EMAIL: aero.sayan@gmail.com
# DATE : 2/JULY/2018
# INFO : How does hot soup sale change in winter based on temperature?
#      : Here, we do linear regression with ordinary least squares

import numpy as np
import matplotlib.pyplot as plt

n = 100                                                   # no. of data points
temp = np.linspace(3, 30, n)                              # temperature (deg C)
noise = np.random.randint(-5,7,size = n)                 # noise to simulate RL
soup = np.linspace(40, 22 , n, dtype = 'int') + noise     # soup sale count

# We are re-assigning the data since we like to write in this form
x = temp                                                  # x co-ordinate
y = soup                                                  # y co-ordinate
x_bar = sum(x)/float(n)                                   # average of x
y_bar = sum(y)/float(n)                                   # average of y


m = sum((x-x_bar)*(y-y_bar))/ float(sum((x-x_bar)**2) )   # slope
b = y_bar - m*x_bar                                       # y intercept

print("The linear regression has resulted in ...")
print("m(slope)       : ",m )
print("b(y intercept) : ",b )


plt.scatter(x,y)
plt.plot(x,m*x + b,'-r')
plt.title("Temperature vs Soup Sale \n (linear regression with ordinary least squares)")
plt.xlabel("Temp in degree Celcius")
plt.ylabel("Hot soup bowls sold")
plt.grid()
plt.show()
