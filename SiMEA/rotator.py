__author__ = 'V_AD'
import scipy.misc
import scipy.ndimage
from numpy import *
from matplotlib.pylab import *



def rotator(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_x = point[0]-centerPoint[0]
    temp_y = point[1] - centerPoint [1]
    dist = sqrt(temp_x**2 + temp_y**2)
    # old_ang = arccos(temp_x/dist)
    # total_ang = old_ang + angle
    new_x = float('%.2f'%(cos(angle)*temp_x - sin(angle)*temp_y + centerPoint[0]))
    new_y = float('%.2f'%(sin(angle)*temp_x + cos(angle)*temp_y + centerPoint [1]))

    # temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    # temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    # temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    temp_point = array([new_x,new_y])
    return temp_point

print rotator((0,0),(1,1),45)
