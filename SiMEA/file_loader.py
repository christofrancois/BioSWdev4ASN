__author__ = 'V_AD'
from numpy import *
import pickle
import matplotlib.pyplot as plt
from close_to_array import *
from brian2 import *

connection_map = pickle.load(open("C:/Users/vafaa/Desktop/results/connectionmap.p","rb"))
# final_result = pickle.load(open("C:/Users/vafaa/Desktop/results/final_result.p","rb"))
# indices_array = pickle.load(open("C:/Users/vafaa/Desktop/results/indices_array.p","rb"))
# all_branches = pickle.load(open("C:/Users/vafaa/Desktop/results/all_branches.p","rb"))
somas = pickle.load(open("C:/Users/vafaa/Desktop/results/somas.p","rb"))
i = pickle.load(open("C:/Users/vafaa/Desktop/results/i.p","rb"))
t = pickle.load(open("C:/Users/vafaa/Desktop/results/t.p","rb"))
closest = close_to_electrode ()
final_i = array([])
final_t = array([])

for num, item in list(enumerate(closest)) :
    idx = where(i==item)
    length = len (idx[0])
    if length == 0 :
        print item
    final_i = append(final_i, num*ones([length]))
    final_t = append (final_t, t[idx])

sim_time = 30*second
figure()
plot(t/ms , i , '.k', ms= 1.2 )
xlabel ("time(s)")
ylabel ("Recording Site")
# yticks([])
title ("Spiking Activity")
xlim((sim_time - 27*second)/ms , sim_time/ms)

show()
print "hi"
