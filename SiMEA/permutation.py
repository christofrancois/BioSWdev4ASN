__author__ = 'V_AD'

from shapely.geometry import LineString
from matplotlib.pyplot import *
from locations import *
from numpy import *
import matplotlib.pyplot as plt
import math as mth
import time

def all_permutations (temp_axons,temp_dends):
    results = array([])
    for a in range(1,len(temp_axons)) :
        if float(temp_axons[a][0]) == float(temp_axons[a][6])+1:
            temp_line1 = LineString([(float(temp_axons[a][2]),float(temp_axons[a][3])),(float(temp_axons[a-1][2]),float(temp_axons[a-1][3]))])
        else:
            target_idx = [p[0] for p in zip(*where(temp_axons == array([temp_axons[a][6]]))) if p[1] == 0][0]
            temp_line1 = LineString([(float(temp_axons[a][2]),float(temp_axons[a][3])),(float(temp_axons[target_idx][2]),float(temp_axons[target_idx][3]))])
        for d in range (1,len(temp_dends)):
            if float(temp_dends[d][0]) == float(temp_dends[d][6])+1:
                temp_line2 = LineString([(float(temp_dends[d][2]),float(temp_dends[d][3])),(float(temp_dends[d-1][2]),float(temp_dends[d-1][3]))])
            else:
                target_idx = [p[0] for p in zip(*where(temp_dends == array([temp_dends[d][0]]))) if p[1] == 0][0]
                temp_line2 = LineString([(float(temp_dends[d][2]),float(temp_dends[d][3])),(float(temp_dends[target_idx][2]),float(temp_dends[target_idx][3]))])
        temp_line = temp_line1.intersection(temp_line2)
        if temp_line :
            results = (results,array(temp_line1.intersection(temp_line2)))
    return len(results) , results # this function returns the number of synapses between a set of dends which are totlaly inside a set a axons as well as the points of connection


##################################################################################




def partial_permutation (temp_axons , temp_dends ):
    results = array([])
    for a in range(1,len(temp_axons)) :
        temp_line1 = LineString([(float(temp_axons[a][2]),float(temp_axons[a][3])),(float(temp_axons[a-1][2]),float(temp_axons[a-1][3]))])
        for d in range (1,len(temp_dends)):
            temp_line2 = LineString([(float(temp_dends[d][2]),float(temp_dends[d][3])),(float(temp_dends[d-1][2]),float(temp_dends[d-1][3]))])
            temp_line = temp_line1.intersection(temp_line2)
            if temp_line :
                results = (results,array(temp_line1.intersection(temp_line2)))
    return len(results[1:]) , results[1:] # this function returns the number of synapses between a set of dends which are totlaly inside a set a axons as well as the points of connection
