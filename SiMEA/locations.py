__author__ = 'V_AD'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import random
import visual
import math as mt
import matplotlib.pyplot as plt
from PIL.BmpImagePlugin import o32
from killer import killer

global r, stop_flag
stop_flag = False
def locations (gridsize,neuronsize,totalcell):
#     gridsize = raw_input('Enter the MEA area in micrometers^2: ')
#     neuronsize = raw_input ('Enter the neuron area in micrometers^2: ')
#     neuronnumber = raw_input ('Enter the total number of neurons : ')
    needed_cells = gridsize / neuronsize
    w = int(mt.sqrt(needed_cells))
    h = w
    global r,stop_flag
    r = -3
    def diamond(w,h,rand = -3,draw = False):
        global r,stop_flag
        def plasma(x, y, width, height, c1, c2, c3, c4):
            newWidth = width / 2
            newHeight = height / 2
            global idx,dots
            if (width > gridSize or height > gridSize):
                #Randomly displace the midpoint!
                midPoint = (c1 + c2 + c3 + c4) / 4 + Displace(rand)
                #Calculate the edges by averaging the two corners of each edge.
                edge1 = (c1 + c2) / 2
                edge2 = (c2 + c3) / 2
                edge3 = (c3 + c4) / 2
                edge4 = (c4 + c1) / 2

                #Do the operation over again for each of the four new grids.
                plasma(x, y, newWidth, newHeight, c1, edge1, midPoint, edge4)
                plasma(x + newWidth, y, newWidth, newHeight, edge1, c2, edge2, midPoint)
                plasma(x + newWidth, y + newHeight, newWidth, newHeight, midPoint, edge2, c3, edge3)
                plasma(x, y + newHeight, newWidth, newHeight, edge4, midPoint, edge3, c4)
            else:
                #This is the "base case," where each grid piece is less than the size of a pixel.
                c = (c1 + c2 + c3 + c4) / 4
    #             dots[idx] = c
    #             idx = idx + 1
        #         print(c)
                if (c>0.5):
                    c = 0
                    if (draw == True):
                        visual.points(pos=[x-(100),c,y-(100)], color=(1,0.31,0.1))
                    dots[idx] = c
                    idx = idx + 1
                else:
                    c= 5
                    if (draw == True):
                        visual.points(pos=[x-(100),c,y-(100)], color=(0.8,1,1))
                    dots[idx] = c
                    idx = idx + 1

        def Displace(num):
            rand = (random.uniform(num, 1) - noise)
        #     print rand
            return rand


        global gridSize, gamma, points, width, height, idx,dots
        random.seed('Albert Einstein was a German theoretical physicist.')

        def reshaper (inpt):
            l= len(inpt)
            if (l>4):
                o1= inpt[0:l/4]
                o2= inpt[l/4:l/2]
                o3= inpt[l/2:3*l/4]
                o4= inpt[3*l/4:]
                temp1 = np.concatenate((reshaper(o1),reshaper(o2)),axis=1)
                temp2 = np.concatenate((reshaper(o4),reshaper(o3)),axis=1)
                temp3 = np.concatenate((temp1,temp2),axis = 0)
                return (temp3)
            else:
                outp = np.zeros((2,2))
                outp[0,0:2] = inpt [0:2]
                outp[1,0:2] = np.fliplr([inpt[2:]])[0]
                return(outp)

        nearest_2_power = 0
        while w>2**nearest_2_power:
            nearest_2_power += 1
        nearest_2 = 2**nearest_2_power
        idx = 0
        width = nearest_2*10
        noise = 0.02 # less noise = higher map
        height = nearest_2*10
        gridSize = 10 # size between pixels
        length = 0
        gridtemp = width
        while gridtemp > gridSize:
            gridtemp = gridtemp/2
            length +=1
#         print(length)
        dots = np.zeros (4**length)

        plasma(0,0, width, height, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    #     print(dots)
        positions = np.zeros ((mt.sqrt(len(dots)),mt.sqrt(len(dots))))

        positions = reshaper(dots)
    #     print (positions)
        cells = sum(sum(positions==5))
#         print (cells,'out of', len(positions)**2)
        return (positions,cells)
    position_final , cells = diamond(w, h)
    position_final_cropped = position_final [0:w,0:w]
    cells = sum(sum(position_final_cropped==5))

    while cells > totalcell:

        r += 0.1
        position_final , cells = diamond (w,h,r)
        position_final_cropped = position_final [0:w,0:w]
        cells = sum(sum(position_final_cropped==5))
        # print(cells)
    else:
        r -= 0.1
        position_final , cells = diamond (w,h,r)
        position_final_cropped = position_final [0:w,0:w]
        cells = sum(sum(position_final_cropped==5))

#     else:
# #         if (stop_flag == False):
# #             stop_flag = True
# #         position_final , cells = diamond (w,h,r,True)
#         position_final , cells = diamond (w,h,r,True)
#         position_final_cropped = position_final [0:w,0:w]
#         cells = sum(sum(position_final_cropped==5))
    # else:
    #     r = rand-0.1
    #     locations (w,h,r)

    # print (cells,'out of', len(position_final_cropped)**2)
    q = position_final_cropped
####################### following four lines removes the extra possible positions
    eliminate  = np.random.randint(cells,size = cells-totalcell)
    all_indices = where(q==5)
    for el in eliminate :
        q[all_indices[0][el]][all_indices[1][el]] = 0
################################################# 17 day killer
    q = killer(q,17)
#################################################

    # for i in range (0,len(q)):
    #     for j in range (0,len(q)):
    #         if (q[i,j]==0):
    #             visual.points(pos=[i,0,j], color=(0.8,1,1))
    #         else:
    #             visual.points(pos=[i,1,j], color=(1,0.31,0.1))
    p_temp = np.where(position_final_cropped == 5)
    p = [array([i[1],j[1]]) for i in list (enumerate(p_temp[0])) for j in list(enumerate(p_temp[1])) if i[0] == j[0]]
    p = [array([pp[1]*(sqrt(neuronsize)),pp[0]*(-sqrt(neuronsize)) ]) for pp in p]
    return p,position_final_cropped



def MEA_locations (gridsize,neuronsize,totalcell):
    p,q = locations(gridsize,neuronsize,totalcell)
    # ax = plt.subplot(111)
    # ax.plot ([-i[1] for i in p],[-i[0] for i in p],'ro')
    real_size = 50
    # origp = np.copy(p)
    # length = sqrt(neuronsize)
    for loc in p :
        i0,i1 = np.random.random(2) * (sqrt(neuronsize))
        loc[0] += i0
        loc[1] -= i1

    return p,q
# p,q = locations(50000,5000, 4)
# p,q = MEA_locations(7840000,5000,1000)

# p2,_ = MEA_locations(7840000,50,1000)
# print (q)
# print(p)
# ax = plt.subplot(111)
# ax.plot ([-i[1] for i in p],[-i[0] for i in p],'ro')
#
# ax.set_ylim([-2800,0])
# ax.set_xlim([0,2800])
#
# plt.show()
# print(np.shape(q))
# p = killer(q,0)
# for i in range (0,len(p)):
#         for j in range (0,len(p)):
#             if (p[i,j]==0):
#                 visual.points(pos=[i,0,j], color=(255/255, 255/255, 255/255))
#             elif (p[i,j]==2.5):
#                 visual.points(pos=[i,1,j], color=(51/255, 102/255, 200/255))
#             else:
#                 visual.points(pos=[i,1,j], color=(255/255 ,204/255 ,10/255))
# print (q)
# for i in range (0,len(q)):
#     for j in range (0,len(q)):
#         if (q[i,j]==0):
#             visual.points(pos=[i,0,j], color=(0.8,1,1))
#         else:
#             visual.points(pos=[i,1,j], color=(1,0.31,0.1))