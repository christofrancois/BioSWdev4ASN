__author__ = 'V_AD'

from shapely.geometry import LineString
from matplotlib.pyplot import *
from locations import *
from numpy import *
import matplotlib.pyplot as plt
import math as mth
import time

from boundry_box import *

def status (str):
    cleaner = ' ' * 100
    print '\r'+ cleaner + '\r' + str,


def branch_finder (all_neurons ) :
    branches = {}
    finder_current = 0
    finder_total = len(all_neurons)
    for idx in range (len(all_neurons)):
        branches ["n%d"%idx] = {}
        branches ["n%d"%idx]['axons'] = {}
        branches ["n%d"%idx]['dends'] = {}
        # temp_1 = [p for p in all_neurons[idx] if ((p[0]!=p[6]+1) and len([q for q in all_neurons[idx] if q[0]==p[6]])!=0)]
        temp_1 = [p for p in all_neurons[idx]['points'] if ((int(p[0])!=int(p[6])+1) and len([q for q in all_neurons[idx]['points'] if q[0]==p[6]])!=0)]
        all_0 = array([int(j[0]) for j in all_neurons[idx]['points']])
        places = [where (all_0 == int(jj[0])) for jj in temp_1]
        counter1 = 0 # this is for axons
        counter2 = 0 # this is for dendrites
        for m1 in range (1,len(temp_1)):
            start_idx = int(places[m1][0]) # start is the first point of branch , begin is the 0 point of branch
            prev_start_idx = int(places[m1-1][0])
            begin_idx = int(all_neurons [idx]['points'][prev_start_idx][6]) -1
            if int(all_neurons[idx]['points'][start_idx][1]) == 2:# this goes for axonal brnaches
                branches ['n%d'%idx]['axons']['br%d'%counter1] = {}
                branches ['n%d'%idx]['axons']['br%d'%counter1]['points'] = vstack([all_neurons [idx]['points'][begin_idx],all_neurons[idx]['points'][prev_start_idx : start_idx]]) # the branch is created and the first point is also considered
                branches ['n%d'%idx]['axons']['br%d'%counter1]['boundry'] = boundry_box (branches ['n%d'%idx]['axons']['br%d'%counter1]['points'])
                counter1+=1
            elif int(all_neurons[idx]['points'][int(places[m1][0])][1]) == 3:
                branches ['n%d'%idx]['dends']['br%d'%counter2] = {}
                branches ['n%d'%idx]['dends']['br%d'%counter2]['points'] = vstack([all_neurons [idx]['points'][begin_idx],all_neurons[idx]['points'][prev_start_idx:start_idx]])
                branches ['n%d'%idx]['dends']['br%d'%counter2]['boundry'] = boundry_box (branches ['n%d'%idx]['dends']['br%d'%counter2]['points'])
                counter2+=1
        finder_current +=1
        finder_perc = float(finder_current*100)/finder_total
        status ("%.2f%% of branch finding completed"%finder_perc)
            # for neu in branches :
            #     # col = random.choice(colors)
            #     # col2 = random.choice(colors)
            #     for p in branches [neu]['axons']:
            #         points = branches [neu]['axons'][p]['points']
            #         for t2 in range (len(points)-1) :
            #             plt.plot ( [points[t2][2],points[t2+1][2]],[points[t2][3],points[t2+1][3]],'k')
            #     for q in branches [neu]['dends']:
            #         points2 = branches [neu]['dends'][q]['points']
            #         for t3 in range ( len(points2) -1 ) :
            #             plt.plot ( [points2[t3][2],points2[t3+1][2]],[points2[t3][3],points2[t3+1][3]],'r')
            # show()
            #
            # if (m1 < len(temp_1)-1) :
            #     # if not places[m1+1]-array([1]) > places[m1]:
            #     #     continue
            #     if all_neurons[idx][int(places[m1][0])][1] == 2: # this goes for axonal brnaches
            #         branches ['n%d'%idx]['axons']['br%d'%counter1] = {}
            #         start_idx = all_neurons [idx][int(places[m1][0])][6] -1  # this finds the starting point of the branch
            #         branches ['n%d'%idx]['axons']['br%d'%counter1]['points'] = vstack([all_neurons [idx][start_idx],all_neurons[idx][int(places[m1][0]):int(places[m1+1][0])]]) # the branch is created and the first point is also considered
            #         branches ['n%d'%idx]['axons']['br%d'%counter1]['boundry'] = boundry_box (branches ['n%d'%idx]['axons']['br%d'%counter1]['points'])
            #         counter1+=1
            #     elif all_neurons[idx][int(places[m1][0])][1] == 3:
            #         branches ['n%d'%idx]['dends']['br%d'%counter2] = {}
            #         start_idx = all_neurons [idx][int(places[m1][0])][6] -1
            #         branches ['n%d'%idx]['dends']['br%d'%counter2]['points'] = vstack([all_neurons [idx][start_idx],all_neurons[idx][int(places[m1][0]):int(places[m1+1][0])]])
            #         branches ['n%d'%idx]['dends']['br%d'%counter2]['boundry'] = boundry_box (branches ['n%d'%idx]['dends']['br%d'%counter2]['points'])
            #         counter2+=1
            #
            # elif all_neurons[idx][int(places[m1][0])][1] == 2:
            #     branches ['n%d'%idx]['axons']['br%d'%counter1] = {}
            #     start_idx = all_neurons [idx][int(places[m1][0])][6] -1
            #     branches ['n%d'%idx]['axons']['br%d'%counter1]['points'] = vstack([all_neurons [idx][start_idx],all_neurons[idx][int(places[m1][0]):]])
            #     branches ['n%d'%idx]['axons']['br%d'%counter1]['boundry'] = boundry_box (branches ['n%d'%idx]['axons']['br%d'%counter1]['points'])
            #     counter1+=1
            # else:
            #     branches ['n%d'%idx]['dends']['br%d'%counter2] = {}
            #     start_idx = all_neurons [idx][int(places[m1][0])][6] -1
            #     branches ['n%d'%idx]['dends']['br%d'%counter2]['points'] = vstack([all_neurons [idx][start_idx],all_neurons[idx][int(places[m1][0]):]])
            #     branches ['n%d'%idx]['dends']['br%d'%counter2]['boundry'] = boundry_box (branches ['n%d'%idx]['dends']['br%d'%counter2]['points'])
            #     counter2+=1
    return branches




