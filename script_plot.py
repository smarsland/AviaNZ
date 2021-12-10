import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt

def int_function(x, L):
  if x<-1/L:
      y1,_=spec.sici(-np.pi*L*x)
      y2, _ = spec.sici(-(np.pi * L * x-np.pi))
      y3, _ = spec.sici(-(np.pi * L * x+np.pi))
      y=1/2 -(1/(2*np.pi))*y1-(1/(4*np.pi))*y2-(1/(4*np.pi))*y3
  elif  x>=-1/L and x<0:
      y1, _ = spec.sici(-np.pi * L * x)
      y2, _ = spec.sici(-(np.pi * L * x - np.pi))
      y3, _ = spec.sici(np.pi * L * x + np.pi)
      y = 1 / 2 - (1 / (2 * np.pi)) * y1 - (1 / (4 * np.pi)) * y2 + (1 / (4 * np.pi)) * y3
  elif  x>=0 and x<1/L:
      y1, _ = spec.sici(np.pi * L * x)
      y2, _ = spec.sici(-(np.pi * L * x - np.pi))
      y3, _ = spec.sici(np.pi * L * x + np.pi)
      y = 1 / 2 + (1 / (2 * np.pi)) * y1 - (1 / (4 * np.pi)) * y2 + (1 / (4 * np.pi)) * y3
  else:
      y1, _ = spec.sici(np.pi * L * x)
      y2, _ = spec.sici(np.pi * L * x - np.pi)
      y3, _ = spec.sici(np.pi * L * x + np.pi)
      y = y = 1 / 2 + (1 / (2 * np.pi)) * y1 + (1 / (4 * np.pi)) * y2 + (1 / (4 * np.pi)) * y3

  return y


x=np.arange(-5,5)
L=64
y=np.zeros((np.shape(x)))
for i in range(len(x)):
    y[i]=int_function(x[i],L)

plt.plot(x,y)
plt.show()