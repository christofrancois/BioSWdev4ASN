from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from numpy import *

# the function that I'm going to plot
def z_func(x,y):
    middle_x = (float(x[0][0] + x[-1][-1]))/2
    middle_y = (float(y[0][0] + y[-1][-1]))/2
    return mlab.bivariate_normal(x, y, 20.0, 20.0, middle_x, middle_y)

# x1 = arange(0,100.0,0.5)
# y1 = arange(0,-100.0,-0.5)
# X1,Y1 = meshgrid(x1, y1) # grid of point
# Z1 = z_func(X1, Y1) # evaluation of the function on the grid


# x2 = arange(10.0,20.0,0.5)
# y2 = arange(10.0,20.0,0.5)
# X2,Y2 = meshgrid(x2, y2) # grid of point
# Z2 = z_func(X2, Y2) # evaluation of the function on the grid

# Z = append(Z1,Z2,axis=1)
# x = append(x1,x2,axis=1)
# y = append(y1,y2,axis=1)
# X,Y = meshgrid(x, y)

def status (str):
    cleaner = ' ' * 100
    print '\r'+ cleaner + '\r' + str,


total_x = arange(0,2800.0,1)
total_y = arange(0,-2800,-1)
total_X,total_Y = meshgrid(total_x, total_y)
Z = zeros_like (total_X)



size_x = 8
size_y = 8
counter = 0
total = size_x*size_y
for j in range(1,size_x+1) :
    for i in range(1,size_y+1) :
        if not ( (i== 1 or i==size_x) and (j==1 or j == size_y  )):
            interval_x = (len(total_x)/(size_x +1))
            interval_y = (len(total_y)/(size_y +1))
            x_temp = total_X [interval_y*j-interval_y/2:interval_y*j+interval_y/2,interval_x*i-interval_x/2:interval_x*i+interval_x/2]
            y_temp = total_Y [interval_x*i-interval_x/2:interval_x*i+interval_x/2,interval_y*j-interval_y/2:interval_y*j+interval_y/2]
            z_temp = z_func(x_temp,y_temp)
            Z[interval_x*i-interval_x/2:interval_x*i+interval_x/2, interval_y*j-interval_y/2:interval_y*j+interval_y/2] = z_temp
            counter+=1
            completed = float(counter) *100 /total
            status("%.2f completed"%completed)




# im = imshow(Z,cmap=cm.RdBu) # drawing the function
# # adding the Contour lines with labels
# cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im) # adding the colobar on the right
# # latex fashion title
# title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
# show()
temp_min = min([p for q in Z for p in q])
temp_max = max([p for q in Z for p in q])
old_ = array([temp_min,temp_max])
new_ = array([0,1])

def aranger (old_, new_, x):
    old_range = old_[1] - old_[0]
    new_range = new_[1] - new_[0]
    new_value = (((x-old_[0])*new_range)/old_range) + new_[0]
    return new_value

Z2 = zeros_like (Z)
for i in range (shape(Z)[0]):
    for j in range(shape(Z)[1]):
        Z2[i,j]  = aranger(old_,new_,Z[i,j])


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(total_X, total_Y, Z,rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zlim ([0,0.002])
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
