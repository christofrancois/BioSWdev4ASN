__author__ = 'V_AD'
from numpy import *
import pickle
import matplotlib.pyplot as plt


def find_nearest (point, point_array ):
    i = pickle.load(open("C:/Users/vafaa/Desktop/results/i.p","rb"))
    t = pickle.load(open("C:/Users/vafaa/Desktop/results/t.p","rb"))
    results = array([])
    for idx,qp in list(enumerate(point_array)) :
        results = append(results,sqrt((point[0]-qp[0])**2 + (point[1]-qp[1])**2 ))
    while len(where(i==int(where(results == min(results)) [0]))[0]) == 0 or len(where(i==int(where(results == min(results)) [0]))[0]) > 500 :
        results = delete (results ,int(where(results == min(results)) [0]) )
    return (int(where(results == min(results)) [0]))

def close_to_electrode () :
    connection_map = pickle.load(open("C:/Users/vafaa/Desktop/results/connectionmap.p","rb"))
    # final_result = pickle.load(open("C:/Users/vafaa/Desktop/results/final_result.p","rb"))
    # indices_array = pickle.load(open("C:/Users/vafaa/Desktop/results/indices_array.p","rb"))
    # all_branches = pickle.load(open("C:/Users/vafaa/Desktop/results/all_branches.p","rb"))
    somas = pickle.load(open("C:/Users/vafaa/Desktop/results/somas.p","rb"))

    elecs_x = range ( 288, 2305, 288)
    elecs_y = range ( -288, -2305, -288)
    grid = meshgrid (elecs_x,elecs_y)




    points = array ([0,0])
    for i in range (8) :
        for j in range (8):
            point = array ( [ grid[0][i][j],grid[1][i][j] ] )
            points = vstack([points,point])
    points = points [1:]

    final_indices = array([])
    for pp in points:
        final_indices = append(final_indices,find_nearest(pp , somas))
    final_indices = delete(final_indices,[0,7,56,63])
    return final_indices
# print "hi"