__author__ = 'V_AD'
from shapely.geometry import LineString
from matplotlib.pyplot import *
from locations import *
from numpy import *
import numpy as np
import numpy as np
from rotator import *
import random
import matplotlib.pyplot as plt
import math as mth
import time
def morpho_generator (area, neuron , n, morpho_source ) :
    p,q = MEA_locations(area,neuron, n)
    print (p)
    # print morpho_source [5]
    total_morpho = zeros ([n,shape(morpho_source)[0],shape(morpho_source)[1]])
    morpho_temp = copy(morpho_source)
    for r in range (n):
        soma_x = float(morpho_temp[0][2]) + p[r][0]
        soma_y = float(morpho_temp[0][3]) + p[r][1]
        angle = np.random.random(1) * 360
        for l in range(len(morpho_temp)):
            temp_x = float(morpho_temp[l][2]) + p[r][0]
            temp_y = float(morpho_temp[l][3]) + p[r][1]
            point  = rotator((soma_x,soma_y),(temp_x,temp_y),angle)
            # morpho_temp[l][2] = str(float(morpho_temp[l][2]) + p[r][0])
            # morpho_temp[l][3] = str(float(morpho_temp[l][3]) + p[r][1])
            morpho_temp[l][2] = str(point[0])
            morpho_temp[l][3] = str(point[1])
            # l[2] = str(float(l[2]) + p[r][1])
            # l[3] = str(float(l[3]) + p[r][0])
        total_morpho [r,:,:] = morpho_temp
        # print morpho_temp[0]
        morpho_temp = copy(morpho_source)

    # print morpho_source[5]
    return total_morpho , p,q



def multi_morpho_generator (area, neuron , n, morpho_source ,types, dist ) :

    p,q = MEA_locations(area,neuron, n)
    n = len(p)
    # print (p)
    # print morpho_source [5]
    pick = array ([])
    for idx , qq in list(enumerate(dist)):
        temp_ = int(qq*n)
        pick = append(pick, (idx)*ones(temp_)) # this creates an array of 1 , 2, ... each repeated by number of cells per type. in next step it will be shuffled to determine the final formation of all type of cells.
    np.random.shuffle(pick)
    while len(pick) < n :
        pick = append (pick, int(rand(1) * (len(dist)-1)))
        random.shuffle(pick)
    total_morpho = {}
    # total_morpho = zeros ([n,shape(morpho_source)[0],shape(morpho_source)[1]])
    # morpho_temp = copy(morpho_source)

    for nm in range (len(dist)):
        print "%d neurons of type %d" % (shape(where(pick==nm))[1],nm)
    for r in range (n):
        morpho_temp = copy(morpho_source[types[int(pick[r])]]) # this put the right model inside the temp
        # morpho_temp = [map(float,p) for p in morpho_temp]
        soma_x = float(morpho_temp[0][2]) + p[r][0]
        soma_y = float(morpho_temp[0][3]) + p[r][1]
        angle = np.random.random(1) * 360
        for l in range(len(morpho_temp)):
            temp_x = float(morpho_temp[l][2]) + p[r][0]
            temp_y = float(morpho_temp[l][3]) + p[r][1]
            point  = rotator((soma_x,soma_y),(temp_x,temp_y),angle)
            # morpho_temp[l][2] = str(float(morpho_temp[l][2]) + p[r][0])
            # morpho_temp[l][3] = str(float(morpho_temp[l][3]) + p[r][1])
            morpho_temp[l][2] = str(point[0])
            morpho_temp[l][3] = str(point[1])

            # l[2] = str(float(l[2]) + p[r][1])
            # l[3] = str(float(l[3]) + p[r][0])
        total_morpho [r] = {}
        total_morpho [r]['type'] = types[int(pick[r])]
        total_morpho [r]['points'] = [map(float,pp) for pp in morpho_temp]

        # total_morpho [r,:,:] = morpho_temp
        # print morpho_temp[0]
        # morpho_temp = copy(morpho_source)

    # print morpho_source[5]
    return total_morpho , p,q , pick
