__author__ = 'V_AD'

from shapely.geometry import LineString
from matplotlib.pyplot import *
from locations import *
from numpy import *
import matplotlib.pyplot as plt
import math as mth
import time

def boundry_box (set):
    xs = [q[2] for q in set]
    x_max = max(xs)
    x_min = min(xs)
    ys = [q[3] for q in set]
    y_max = max(ys)
    y_min = min(ys)
    return array([x_min,y_max]),array([x_max,y_max]),array([x_max,y_min]),array([x_min,y_min])




