__author__ = 'V_AD'
from matplotlib.pyplot import  *
from pylab import *
from numpy import *
import pickle

with open ('c:/Users/vafaa/PycharmProjects/Cox_Cuda/Xs','rb') as h1:
    Xs = pickle.load(h1)
with open ('c:/Users/vafaa/PycharmProjects/Cox_Cuda/Ys','rb') as h2:
    Ys = pickle.load(h2)
with open ('c:/Users/vafaa/PycharmProjects/Cox_Cuda/Ss','rb') as h3:
    Ss = pickle.load(h3)
fig = figure ( )
# ax = fig.gca()
ax = fig.add_subplot(111)
ax.set_xticks(arange(1.5,34.5,1))
ax.set_yticks(arange(1.5,34.5,1))
# ax.set_xticks(arange(1,35,1))
# ax.set_yticks(arange(1,35,1))
scatter (Xs,Ys,s=Ss)
ax.set_xlim([0.5,34.5])
ax.set_ylim([0.5,34.5])
grid(True)
ax.set_aspect('equal')
show()