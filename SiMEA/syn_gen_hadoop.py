__author__ = 'V_AD'
from boto import connect_autoscale

__author__ = 'V_AD'

from numpy import *
import pickle
import matplotlib.pyplot as plt
import csv

def syn_generator ():
    #############################
    #############################
    ############################# fix the addresses to the files I sent you before
    #############################
    #############################
    # results = load('C:/Users/andalibi/Local/final_result.npy')
    connection_map = pickle.load(open("C:/Users/vafaa/Desktop/results/2_neurons/connectionmap.p","rb"))
    final_result = pickle.load(open("C:/Users/vafaa/Desktop/results/2_neurons/final_result.p","rb"))
    indices_array = pickle.load(open("C:/Users/vafaa/Desktop/results/2_neurons/indices_array.p","rb"))
    all_branches = pickle.load(open("C:/Users/vafaa/Desktop/results/2_neurons/all_branches.p","rb"))
    #############################
    #############################
    #############################
    #############################


    # all_neurons = pickle.load( open( "C:/Users/vafaa/Desktop/results/all_neurons.p", "rb" ) )
    #this fille returns the arrays to be used in following syntaxes :
    # S = Synapses(P, Q, pre='v += w')
    # S.connect([1, 2], [1, 2],n=2)
    # if there are three types of neurons (3 NeuronGroup) then it will be at most 6 groups of synapses , so 12 arrays to be returnd

    from_num = unique([final_result[p]['from']['type'] for p in final_result])
    to_num = unique([final_result[p]['to']['type'] for p in final_result])

    synapse_syntax  = array([])
    for q in from_num :
        for p in to_num :
            synapse_syntax = append(synapse_syntax, 'S%d%d = Synapses(N%d,N%d, stdp%d, pre=pre%d, post=post%d )'%(int(q),int(p),int(q),int(p),int(q),int(q),int(q)))
    from_arr = {}
    to_arr = {}
    from_arr['00'] = array([])
    from_arr['01'] = array([])
    from_arr['10'] = array([])
    from_arr['11'] = array([])
    to_arr['00'] = array([])
    to_arr['01'] = array([])
    to_arr['10'] = array([])
    to_arr['11'] = array([])

    connect_syntax = array([])
    # counter = 0
    for q2 in final_result :
        if final_result[q2]['n'] > 0 :
            from_type = int(final_result[q2]['from']['type'])
            to_type = int(final_result[q2]['to']['type'])
            from_temp =  int(final_result[q2]['from']['idx'][1:])
            to_temp  = int(final_result[q2]['to']['idx'][1:])
            from_idx = indices_array [from_type][from_temp]
            to_idx =  indices_array [to_type][to_temp]
            connect_syntax = append(connect_syntax, 'S%d%d.connect (%d,%d ,n=%d ) '%(from_type,to_type, from_idx , to_idx ,int(final_result[q2]['n'])))
            # connect_syntax = append(connect_syntax, 'S%d%d.connect (%d,%d ) '%(from_type,to_type, from_idx , to_idx ))
            # from_arr['%d%d'%(from_type,to_type)] = append(from_arr['%d%d'%(from_type,to_type)], from_idx)
            # to_arr['%d%d'%(from_type,to_type)] = append (to_arr['%d%d'%(from_type,to_type)] , to_idx)
            # counter +=1
            if (from_type == 1 and to_type ==0 ):
                connect_syntax = append(connect_syntax, 'S%d%d.w =  1e-2'%(from_type,to_type))
    syn1 = 'S00.connect (%s,%s,p=0.2)'%(from_arr['00'],to_arr['00'])
    syn2 = 'S01.connect (%s,%s,p=0.2)'%(from_arr['01'],to_arr['01'])
    syn3 = 'S10.connect (%s,%s,p=0.2)'%(from_arr['10'],to_arr['10'])
    syn4 = 'S11.connect (%s,%s,p=0.2)'%(from_arr['11'],to_arr['11'])
    # syn1 = "S00.connect (" + str([int(pp) for pp in from_arr['00']]) + "," + str([int(pp2) for pp2 in to_arr['00']]) + ",p=0.9)"
    # syn2 = "S01.connect (" + str([int(pp) for pp in from_arr['01']]) + "," + str([int(pp2) for pp2 in to_arr['01']]) + ",p=0.9)"
    # syn3 = "S10.connect (" + str([int(pp) for pp in from_arr['10']]) + "," + str([int(pp2) for pp2 in to_arr['10']]) + ",p=0.9)"
    # syn4 = "S11.connect (" + str([int(pp) for pp in from_arr['11']]) + "," + str([int(pp2) for pp2 in to_arr['11']]) + ",p=0.9)"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    print "plotting the neurons"
    branch_status = 0
    branch_total = len([ne for ne in all_branches])
    for neu in all_branches :
        # col = random.choice(colors)
        # col2 = random.choice(colors)
        for p in all_branches [neu]['axons']:
            points = all_branches [neu]['axons'][p]['points']
            for t2 in range (len(points)-1) :
                plt.plot ( [points[t2][2],points[t2+1][2]],[points[t2][3],points[t2+1][3]],'k')
        for q in all_branches [neu]['dends']:
            points2 = all_branches [neu]['dends'][q]['points']
            for t3 in range ( len(points2) -1 ) :
                plt.plot ( [points2[t3][2],points2[t3+1][2]],[points2[t3][3],points2[t3+1][3]],'r')
    #
    for_plot1 = [i for i in [final_result[p]['points'] for p in final_result] if len(i)!= 1]
    for_plot = [qq for jj in for_plot1 for qq in jj]
    ##############
    ############## un-comment following line for plotting my result
    # plt.plot([i[0] for i in for_plot],[i[1] for i in for_plot],'b^')
    ##############
    ##############
    print (final_result)
    plt.axis('equal')

    #####################
    ##################### following lines are used to plot your results, note the file name and address
    #####################
    with open("C:/Users/vafaa/Desktop/results/2_neurons/result1000.csv","rb") as csvfile :
        reader = csv.reader(csvfile, delimiter='\t')
        qp = [line for line in csvfile]
        X = [float(l.split("\t(")[1].split(',')[0]) for l in qp]
        Y = [ float(l.split("\t(")[1].split(', ')[1].split(')\r')[0]) for l in qp]
    plt.plot(X,Y,'b^')
    #####################
    #####################
    #####################


    plt.show()
    return synapse_syntax,connect_syntax
    # return syn1,syn2,syn3,syn4


syn_generator()
# print"hi"
