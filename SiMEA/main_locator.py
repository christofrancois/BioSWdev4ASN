__author__ = 'V_AD'
from shapely.geometry import LineString
from matplotlib.pyplot import *
from locations import *
from numpy import *
import matplotlib.pyplot as plt
import math as mth
import time
from intersect_finder import *
from boundry_box import *
from branch_finder import *
from morpho_generator import *
from permutation import *
import pickle
####################################################
def status (str):
    cleaner = ' ' * 100
    print '\r'+ cleaner + '\r' + str,
####################################################

NN = 1000
neuron_size = 5000

plate_size = 7480000
# neuron = array(['1', '1', '2.39', '-1.6', '-563.94', '11.1795', '-1'])
types = ['P_D27B','P_D20D','P_I03481','P_C300898C-P2','P_C031097B-P3','B_C070600A4','B_C010600C1','B_C010600A2','B_BE23B','B_BE49B']
types_location = ['C:/Users/admin_tunnus/Desktop/P_D27B.swc','C:/Users/admin_tunnus/Desktop/P_D20D.swc','C:/Users/admin_tunnus/Desktop/P_I03481.swc','C:/Users/admin_tunnus/Desktop/P_C300898C-P2.swc','C:/Users/admin_tunnus/Desktop/P_C031097B-P3.swc','C:/Users/admin_tunnus/Desktop/B_C070600A4.swc','C:/Users/admin_tunnus/Desktop/B_C010600C1.swc','C:/Users/admin_tunnus/Desktop/B_C010600A2.swc','C:/Users/admin_tunnus/Desktop/B_BE23B.swc','C:/Users/admin_tunnus/Desktop/B_BE49B.swc' ]
neurons_source = {}
for i in range (len(types)) :
    neurons_source[types[i]] =  array(['1', '1', '2.39', '-1.6', '-563.94', '11.1795', '-1'])
    with open (types_location[i], 'r') as f :
        for line in f:
            if  not (line.startswith('#')):
                neurons_source[types[i]] = vstack ((neurons_source[types[i]], line.split()))
    neurons_source[types[i]] = neurons_source[types[i]][1:]
# neuron = neuron [1:]
dist = array([0.14,0.14,0.14,0.14,0.14,0.06,0.06,0.06,0.06,0.06])
all_neurons,somas, structure , type_array = multi_morpho_generator(plate_size,neuron_size,NN,neurons_source,types,dist)
# somas = divide (somas,neuron_size)
NN  = len(somas)
print ("number of survivals: %d"%NN)
axons = {}
dends = {}
points = {}
for idx in range (NN):
    axons["a%d" %idx] = [p for p in all_neurons[idx]['points'] if p[1]==2]
    dends["d%d" %idx] = [p for p in all_neurons[idx]['points'] if p[1]==3]
    points["n%d" %idx] = {}
# total_max = max(max([ qq[2] for pp in all_neurons for qq in pp ]), max([ q[3] for p in all_neurons for q in p ]))
# total_min = min(min([ qq[2] for pp in all_neurons for qq in pp ]), min([ q[3] for p in all_neurons for q in p ]))
print "starting finding the branches"
all_branches = branch_finder (all_neurons)
print"branches are ready "


colors = array(['b','g','r','c','m','y','k'])

# for neu in all_branches :
#     col = random.choice(colors)
#     col2 = random.choice(colors)
#     for p,q in zip(all_branches [neu]['axons'],all_branches [neu]['dends']):
#         points = all_branches [neu]['axons'][p]['points']
#         points2 = all_branches [neu]['dends'][q]['points']
#         for t2 in range (len(points)-1) :
#             plt.plot ( [points[t2][2],points[t2+1][2]],[points[t2][3],points[t2+1][3]],'k')
#         for t3 in range ( len(points2) -1 ) :
#             plt.plot ( [points2[t3][2],points2[t3+1][2]],[points2[t3][3],points2[t3+1][3]],'r')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# print "plotting the neurons"
# branch_status = 0
# branch_total = len([ne for ne in all_branches])
# for neu in all_branches :
#     # col = random.choice(colors)
#     # col2 = random.choice(colors)
#     for p in all_branches [neu]['axons']:
#         points = all_branches [neu]['axons'][p]['points']
#         for t2 in range (len(points)-1) :
#             plt.plot ( [points[t2][2],points[t2+1][2]],[points[t2][3],points[t2+1][3]],'k')
#     for q in all_branches [neu]['dends']:
#         points2 = all_branches [neu]['dends'][q]['points']
#         for t3 in range ( len(points2) -1 ) :
#             plt.plot ( [points2[t3][2],points2[t3+1][2]],[points2[t3][3],points2[t3+1][3]],'r')
#     branch_status += 1
#     branch_perc = float(branch_status*100)/branch_total
#     status ("%.2f%% completed"%branch_perc)



### creating indices_array :
indices_array = type_array
for pp in range(len (types)):
    temp_ar = zeros ([len(type_array)])
    counter = 0
    for ii, q in list(enumerate(type_array)) :
        if q == pp :
            temp_ar[ii] = counter
            counter +=1
    indices_array = vstack([indices_array,temp_ar])
indices_array =indices_array[1:]






# axis([total_min, total_max,total_min,total_max])
# plt.axis('equal')
# show()
# raw_input("here")



# axons = [q for p in all_neurons for q in all_neurons[p] if q[1] == '2']
# dends = [p for p in all_neurons if p[1] == '3']

# fig = figure()
# ax = fig.add_subplot(111)
# line = LineString ([(0,0),(1,1)])
# plot_coords(ax, line)
# plot_bounds(ax, line)
# plot_line(ax, line)
# show()
final_result = {}
# points = array([0,0])
# passed_number = 0.
# sh1 = shape ([axons[p] for p in axons])
# sh2 = shape ([dends[p] for p in dends])
# total_number = sh1[0] * (sh1[1]-1) * (sh2[0]-1) * (sh2[1]-1)
# print (total_number)
fr = array([])
to = array([])
sy = array([])

print "finding boundry boxes"
axon_borders = zeros ([NN,4,2]) # border points of each neuron 0 is upper left, 1 is upper right and so on
dend_borders = zeros ([NN,4,2])
counter1 = 0
for neuron in range (NN):
    axon_borders[neuron,0,:],axon_borders[neuron,1,:],axon_borders[neuron,2,:], axon_borders[neuron,3,:] = boundry_box (axons['a%d'%neuron])
    dend_borders[neuron,0,:] , dend_borders[neuron,1,:] , dend_borders[neuron,2,:], dend_borders[neuron,3,:] =  boundry_box (dends['d%d'%neuron])
    counter1+= 1
    percentage = float(counter1)/NN *100
    status ("%.2f %% of locating borders completed\n" %percentage)
print (structure)
# print (axon_borders)
# print(dend_borders)
### next two lines draw the whole boundries
# plt.plot([qq[0] for pp in axon_borders for qq in pp],[qq[1] for pp in axon_borders for qq in pp],'ro')
# plt.plot([qq[0] for pp in dend_borders for qq in pp],[qq[1] for pp in dend_borders for qq in pp],'bs')
# show()
# raw_input("press to continue...")
dig_level = 1
zeros_src = 0
syn_set = -1

connection_map = {}


print "start finding axons and dends borders"
# for q,t in zip(dend_borders,axon_borders) :
#     col = random.choice(colors)
#     for t2 in range (3) :
#         plt.plot ( [t[t2][0],t[t2+1][0]],[t[t2][1],t[t2+1][1]],col)
#         plt.plot([q[t2][0],q[t2+1][0]],[q[t2][1],q[t2+1][1]],col )
#     plt.plot ( [t[0][0],t[3][0]],[t[0][1],t[3][1]],col)
#     plt.plot ( [q[0][0],q[3][0]],[q[0][1],q[3][1]],col)
# show()

print "creating connection map "
map_current = 0
map_total = NN
for targ in range (NN) :
    temp_axon_border = axon_borders[targ,:,:]
    connection_map['n%d'%targ] = {}
    connection_map['n%d'%targ]['neuron_type'] = type_array[targ]
    for ref in range (NN):
        if targ!=ref :
            temp_dends_border = dend_borders[ref,:,:]
            _,temp_type,temp_boundry = intersect_finder (temp_axon_border[0],temp_axon_border[1],temp_axon_border[2],temp_axon_border[3],temp_dends_border[0],temp_dends_border[1],temp_dends_border[2],temp_dends_border[3],boundry_flag=1)
            if temp_type != '0P':
                connection_map['n%d'%targ]['n%d'%ref]= {}
                connection_map['n%d'%targ]['n%d'%ref]['type'] = temp_type
                connection_map['n%d'%targ]['n%d'%ref]['neuron_type'] = type_array[ref]
                connection_map['n%d'%targ]['n%d'%ref]['boundry'] = temp_boundry
    map_current +=1
    map_perc = float (map_current *100 ) / map_total
    status ("%.2f%% of creating connection map completed"%map_perc)


# show( )
# print (connection_map)
# raw_input("somehting")
# show()
# print connection_map


total_calc =  len([q for p in connection_map for q in connection_map[p] if q!=  'neuron_type'])
current_calc = 0
print "Finding Connections"
for src , target in list(enumerate(connection_map)):
    for dest, reference in list(enumerate(connection_map[target])):
        if reference != 'neuron_type' :
            print target,reference
            start_time = time.time()
            temp_boundries_boundry = connection_map[target][reference]['boundry']
            temp_type = connection_map[target][reference]['type']
            print temp_type
            temp_axons_branches = [all_branches[target]['axons'][b] for b in all_branches[target]['axons'] if intersect_finder\
                (all_branches[target]['axons'][b]['boundry'][0],all_branches[target]['axons'][b]['boundry'][1],\
                 all_branches[target]['axons'][b]['boundry'][2],all_branches[target]['axons'][b]['boundry'][3],\
                 temp_boundries_boundry[0],temp_boundries_boundry[1],temp_boundries_boundry[2],temp_boundries_boundry[3])[1] != '0P' ]
            temp_dends_branches = [all_branches[reference]['dends'][b] for b in  all_branches[reference]['dends'] if intersect_finder\
                (all_branches[reference]['dends'][b]['boundry'][0],all_branches[reference]['dends'][b]['boundry'][1],\
                 all_branches[reference]['dends'][b]['boundry'][2],all_branches[reference]['dends'][b]['boundry'][3],\
                 temp_boundries_boundry[0],temp_boundries_boundry[1],temp_boundries_boundry[2],temp_boundries_boundry[3])[1] != '0P' ]
            syn_set+=1
            final_result['syn_set%d'%syn_set] = {}
            final_result['syn_set%d'%syn_set]['from'] = {}
            final_result['syn_set%d'%syn_set]['from']['idx'] = target
            final_result['syn_set%d'%syn_set]['from']['type'] =  connection_map[target]['neuron_type']
            final_result['syn_set%d'%syn_set]['to'] = {}
            final_result['syn_set%d'%syn_set]['to']['idx'] = reference
            final_result['syn_set%d'%syn_set]['to']['type'] =connection_map[target][reference]['neuron_type']
            final_result['syn_set%d'%syn_set]['n'] = 0
            final_result['syn_set%d'%syn_set]['points'] = array ([0,0]) # remeber to remove first element since it's an empty array
            total_per = len(temp_axons_branches) * len(temp_dends_branches)
            current_per = 0
            for t_ax in temp_axons_branches :
                for t_de in temp_dends_branches:
                    if (intersect_finder(t_ax['boundry'][0],t_ax['boundry'][1],t_ax['boundry'][2],t_ax['boundry'][3],t_de['boundry'][0],t_de['boundry'][1],t_de['boundry'][2],t_de['boundry'][3])[1] != '0P'):
                        temp_final_n  ,temp_final_points = partial_permutation([q1 for q1 in t_ax['points']],[q2 for q2 in t_de['points']])
                        final_result['syn_set%d'%syn_set]['n'] += temp_final_n
                        if temp_final_n != 0 :
                            final_result['syn_set%d'%syn_set]['points'] =  vstack ([final_result['syn_set%d'%syn_set]['points'],temp_final_points])
                        # status ("%d found"%final_result['syn_set%d'%syn_set]['n'])
                    current_per +=  1
                    current_perc= float(current_per)*100/total_per
                    status("%.2f%% of current neuron is finnished "%current_perc)
            elapsed_time = time.time() - start_time
            print (" the time is : %f " %elapsed_time)
            final_result['syn_set%d'%syn_set]['points'] = final_result['syn_set%d'%syn_set]['points'][1:]
            current_calc += 1
            total_calc_perc=  float(current_calc)*100 / total_calc
            status ("#################### Totally %.2f%% completed" %total_calc_perc)

# print(final_result)

# following three lines gather the points of connection and plot them with green triangles
# for_plot1 = [i for i in [final_result[p]['points'] for p in final_result] if len(i)!= 1]
# for_plot = [qq for jj in for_plot1 for qq in jj]
# plt.plot([i[0] for i in for_plot],[i[1] for i in for_plot],'b^')
# plt.plot([i[0] for i in for_plot],[i[1] for i in for_plot],'y^')
#########
# save('C:/Users/andalibi/Local/connectionmap', connection_map)
# save('C:/Users/andalibi/Local/final_result', final_result)


pickle.dump( connection_map, open( "C:/Users/admin_tunnus/Desktop/results/connectionmap.p", "wb" ) )
pickle.dump( final_result, open( "C:/Users/admin_tunnus/Desktop/results/final_result.p", "wb" ) )
pickle.dump( indices_array, open( "C:/Users/admin_tunnus/Desktop/results/indices_array.p", "wb" ) )
pickle.dump( all_branches, open( "C:/Users/admin_tunnus/Desktop/results/all_branches.p", "wb" ) )
pickle.dump( all_neurons, open( "C:/Users/admin_tunnus/Desktop/results/all_neurons.p", "wb" ) )
pickle.dump( somas, open( "C:/Users/admin_tunnus/Desktop/results/somas.p", "wb" ) )
pickle.dump( structure, open( "C:/Users/admin_tunnus/Desktop/results/structure.p", "wb" ) )
# plt.axis('equal')
# show()

