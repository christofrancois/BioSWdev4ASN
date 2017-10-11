__author__ = 'V_AD'

from shapely.geometry import LineString
from matplotlib.pyplot import *
from locations import *
from numpy import *
import matplotlib.pyplot as plt
import math as mth
import time


def intersect_finder (UL1,UR1, DR1,DL1,UL2,UR2, DR2,DL2,boundry_flag = 0): # boundry_flag defines if you need the boundry between the squares or not , if set to 1, it will return the boundry as well
    types = ['0P','2P_2L','2P_1L', '4P_2L' ]
    points = array([0,0])
    lines1 = [LineString for zz in range (4)]
    lines2 = [LineString for zz in range (4)]
    if UL1[0]<=UL2[0] and UL1[1]>=UL2[1] and UR1[0]>=UR2[0] and UR1[1]>=UR2[1] and DR1[0]>=DR2[0] and DR1[1]<=DR2[1] and DL1[0]<=DL2[0]and DL1[1]<=DL2[1] :
        # print "dendrite box is  inside axon box !"
        if boundry_flag == 0 :
            return {},'inside'
        else:
            return {},'inside' , array([UL2,UR2, DR2,DL2])
    if UL2[0]<=UL1[0] and UL2[1]>=UL1[1] and UR2[0]>=UR1[0] and UR2[1]>=UR1[1] and DR2[0]>=DR1[0] and DR2[1]<=DR1[1] and DL2[0]<=DL1[0]and DL2[1]<=DL1[1] :
        # print "Axon box is inside dendrite box !"
        if boundry_flag == 0 :
            return {},'outside'
        else:
            return {}, 'outside' , array([UL1,UR1, DR1,DL1])
    lines1[0] = LineString([UL1,UR1])
    lines1[1] = LineString([UR1,DR1])
    lines1[2] = LineString([DR1,DL1])
    lines1[3] = LineString([DL1,UL1])

    lines2[0] = LineString([UL2,UR2])
    lines2[1] = LineString([UR2,DR2])
    lines2[2] = LineString([DR2,DL2])
    lines2[3] = LineString([DL2,UL2])
    out = {}
    touch_flag = 0 # this will become one if there is any touching between the edges
    counter = 0
    boundry = array([])
    for first , l1 in list(enumerate(lines1)):
        for second, l2 in list(enumerate(lines2)):
            temp = l1.intersection(l2)
            if (temp) and shape(temp) != (2,2):
                assert (len(array(temp))<=2),"more than one point between two lines"
                # points = vstack ((points,array(temp)))
                out[counter] = {}
                out[counter]['point'] = array(temp)
                out[counter]['line1'] = first
                out[counter]['line2'] = second
                counter+= 1

    # if touch_flag :
    #     shape_arr = [shape(out[pp]['point']) for pp in out]
    #     l = len( shape_arr )
    #     idx = array([])
    #     for qp in range (l) :
    #         if shape_arr [qp] != (2,):
    #             idx = append(idx,qp) # idx is the indices of the touch lines
    qq = array([out[p]['line1'] for p in out])

    length = len(qq) - len(unique(qq))
    type = types[0]
    if length == 1 :
        type = types[2]
    elif length == 2 :
        type = types [3]
    elif len (qq):
        # assert (len (qq)== 2) , "the type is 2P_2L but the intersectoin are not equal to 2 (either more or less)"
        type = types [1]



    assert (len(points) <= 4 ), "There are more than four intersections"
    assert (len(points) != 3) , "Warning! there are three points "
    assert (len(points) != 1 ), "there is just one intersecting point ?!"

    if boundry_flag: # here it finds the common boundry and return it
        if type == types[1] :
            if out[0]['line1'] ==0 :
                if out[0]['line2'] ==1 :
                    boundry = array ([ UL1,out[0]['point'],DR2,out[1]['point'] ])
                    return out,type, boundry
                else:
                    boundry = array ([ out[0]['point'],UR1,out[1]['point'],DL2 ])
                    return out,type, boundry
            elif out[0]['line1'] ==1 :
                if out[0]['line2'] ==0 :
                    boundry = array ([ UL2,out[0]['point'],DR1,out[1]['point'] ])
                    return out,type, boundry
                else:
                    boundry = array ([ out[1]['point'], UR1,out[0]['point'],DL2 ])
                    return out,type, boundry
            elif out[0]['line1'] ==2 :
                if out[0]['line2'] ==1 :
                    boundry = array ([ out[1]['point'], UR2,out[0]['point'],DL1 ])
                    return out,type, boundry
                else:
                    boundry = array ([ UL2, out[1]['point'], DR1,out[0]['point'] ])
                    return out,type, boundry
            else:
                if out[0]['line2'] ==0 :
                    boundry = array ([ out[0]['point'],UR2,out[1]['point'],DL1 ])
                    return out,type, boundry
                else:
                    boundry = array ([UL1, out[0]['point'],DR2,out[1]['point'] ])
                    return out,type, boundry
        elif type == types[2]:
            if out[0]['line1'] ==0:
                boundry = array ([ out[1]['point'],out[0]['point'], DR2,DL2 ])
                return out,type, boundry
            elif out[0]['line1'] ==1 :
                boundry = array ([ UL2, out[0]['point'],out[1]['point'], DL2 ])
                return out,type, boundry
            elif out[0]['line1'] ==2 :
                boundry = array ([ UL2,UR2, out[0]['point'],out[1]['point'] ])
                return out,type, boundry
            else:
                boundry = array ([  out[0]['point'],UR2,DR2,out[1]['point'] ])
                return out,type, boundry
        elif type == types[3]:
            if out[0]['line1'] ==0:
                boundry = array ([ out[1]['point'],out[0]['point'],out[2]['point'],out[3]['point']  ])
                return out,type, boundry
            elif out[0]['line1'] ==1:
                boundry = array ([ out[2]['point'],out[3]['point'],out[0]['point'],out[1]['point']  ])
                return out,type, boundry
        elif type == types[0]:
            boundry = array ([])
            return out,type, boundry

        else:
            print "error , intersection type is unknown!"

    else:
        return out, type

