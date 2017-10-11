# converting the multiple type of neurons to 2

from numpy import *
import pickle


def to_2 ():
    each_type = array ([69,70,70,69,69,29,29,29,30,29])
    synapse_syntax = pickle.load(open("C:/Users/vafaa/Desktop/results/synapse_syntax.p","rb"))
    connect_syntax = pickle.load(open("C:/Users/vafaa/Desktop/results/connect_syntax.p","rb"))
    syntax_final = array([])
    for connection in connect_syntax :
        if connection.split('=')[1] != '  1e-2' :
            from_syn = 0 if int (connection[1]) < 5 else 1
            to_syn = 0 if int (connection[2]) < 5 else 1
            from_offset = int(connection.split("(")[1].split(',')[0])
            to_offset = int(connection.split(",")[1])
            if from_syn == 0 :
                from_base = sum(each_type[:int(connection[1])])
            else :
                from_base = sum(each_type[5:int(connection[1])])
            if to_syn ==0 :
                to_base = sum(each_type[:int(connection[2])])
            else :
                to_base = sum(each_type[5:int(connection[2])])
            from_idx  = from_base + from_offset
            to_idx = to_base + to_offset
            syntax_final = append(syntax_final, 'S%d%d.connect (%d,%d ,n=%d ) '%(from_syn,to_syn, from_idx , to_idx ,int(connection.split('=')[1].split(')')[0])))
        # else:
        #     from_syn = 0 if int (connection[1]) < 5 else 1
        #     to_syn = 0 if int (connection[2]) < 5 else 1
    # print "hi"
    return syntax_final

# to_2()