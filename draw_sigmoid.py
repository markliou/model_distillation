import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus']=False
 
 
def  sigmoid(x, T=1):
    return 1.0 / (1.0 + np.exp(-x/T))
 
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
 
x = np.linspace(-10, 10)
y_1 = sigmoid(x)
y_15 = sigmoid(x, 1.5)
y_2 = sigmoid(x, 2)
y_25 = sigmoid(x, 2.5)
y_3 = sigmoid(x, 3)
y_100 = sigmoid(x, 10)

 
plt.xlim(-11,11)
plt.ylim(-1.1,1.1)
 
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
 
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-10,-5,0,5,10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([-1,-0.5,0.5,1])
 
plt.plot(x,y_1,label="T=1",color = "blue")
plt.plot(x,y_15,label="T=1.5", color = "red")
plt.plot(x,y_2,label="T=2", color = "green")
plt.plot(x,y_25,label="T=2.5", color = "pink")
plt.plot(x,y_3,label="T=3", color = "yellow")
plt.plot(x,y_100,label="T=10", color = "gray")
plt.legend()
plt.show()
