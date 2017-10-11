__author__ = 'V_AD'

from numpy import *
import pickle


connection_map = pickle.load(  open( "C:/Users/admin_tunnus/Desktop/results/connectionmap.p", "rb" ) )
final_result = pickle.load(  open( "C:/Users/admin_tunnus/Desktop/results/final_result.p", "rb" ) )
indices_array = pickle.load(  open( "C:/Users/admin_tunnus/Desktop/results/indices_array.p", "rb" ) )
all_branches = pickle.load(  open( "C:/Users/admin_tunnus/Desktop/results/all_branches.p", "rb" ) )
all_neurons = pickle.load(  open( "C:/Users/admin_tunnus/Desktop/results/all_neurons.p", "rb" ) )
somas = pickle.load(  open( "C:/Users/admin_tunnus/Desktop/results/somas.p", "rb" ) )
structure = pickle.load(  open( "C:/Users/admin_tunnus/Desktop/results/structure.p", "rb" ) )


print "hi"


final = zeros([6])

for f1 in all_branches:
    first = f1[1:]
    for num,f2 in list(enumerate(all_branches[str(f1)])):
        second = num
        for br in all_branches [str(f1)][str(f2)] :
            row_num = len ( [ qp for qp in  all_branches [str(f1)][str(f2)][str(br)]['points']] )
            for idx in range ( 0,row_num -1,2 ):
                third = all_branches [str(f1)][str(f2)][str(br)]['points'][idx][2]
                fourth = all_branches [str(f1)][str(f2)][str(br)]['points'][idx][3]
                fifth = all_branches [str(f1)][str(f2)][str(br)]['points'][idx+1][2]
                sixth = all_branches [str(f1)][str(f2)][str(br)]['points'][idx+1][3]
                final = vstack ((final, [int(first),int(second),float(third),float(fourth),float(fifth),float(sixth)]))
final = final [1:]
savetxt("C:/Users/admin_tunnus/Desktop/results/foo.csv",final, delimiter="    ", fmt = '%d  %d  %f  %f  %f  %f')