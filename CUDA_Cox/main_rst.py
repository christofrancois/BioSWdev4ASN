from numpy import *
from datetime import *
from Cox_Method import Cox_Method

def main_rst(data_path,nn):
    '''
    This is the main function that applies the cox method on the rst data prodcued by ELIF simulator that is based on \
    ELIF model (the software can be found from the following web-site: http://www.tech.plymouth.ac.uk/infovis.) This \
    function plots the resulted connectivity map.

    :param data_path: The path to the .rst data file.
    :param nn: Total number of neurons in the network.

    Main internal variables:

    * p: Number of reference neurons in the network.
    * betahats: Matrix of betahat values in the network.
    * betacis: Matrix of confidence interval of betahat values in the network.
    * target:  Target spike train
    * maxi: The length of reference spike trains varies. This parameter defines the length of the longest reference spike train.
    * tsp: The matrix of reference spike trains. The size of this matrix is (maxi x nn-1) where each column corresponds\
    to a reference spike train and in case the length of the spike train is shorter than "maxi", it should be padded by zeros.
    * to_r: The output matrix of the main_rst function expressing the connections and corresponding reference and target neurons. 
    '''
    p = nn - 1 # Number of reference neurons in the network
    betahats = zeros((nn,nn)) # Matrix of betahat values in the network
    betacis = zeros ((nn*3-1 , nn)) # Matrix of confidence interval of betahat values in the network.
    for neuron in range (0,nn):
        with open (data_path ,'rb') as f:
            a = f.read().split()
            a = map(int,a)
            each = (len(a)+1)/nn
            target = a[neuron*each:neuron*each+each] # Target spike train
            target = nonzero(target)[0]
            target = target + 1
            lengths = zeros (nn)
            lengths_b = zeros (nn)
            for i in range(0,nn):
                lengths[i] = len(nonzero(a[(i*each):(i+1)*each])[0])
            if (neuron == 0):
                print ("Length of each train in average: ")
                print(lengths.astype(int))
                print(average(lengths))
            if (neuron == 0):
                maxi = max(lengths[1:])
            elif (neuron == p):
                maxi = max(lengths[0:nn])
            else:
                maxi = max([max(lengths[0:neuron]),max(lengths[neuron+1:])])
            tsp = zeros((maxi,nn-1))-1 #The matrix of reference spike trains. The size of this matrix is (maxi x nn-1) where each column corresponds to a reference spike train and in case the length of the spike train is shorter than "maxi", it should be padded by zeros.
            if (neuron==0):
                for i in range (0,p):
                    tsp[0:lengths[i+1],i] = nonzero(a[((i+1)*each):(i+2)*each])[0]+1
            elif (neuron == p):
                for i in range (0,p):
                    tsp[0:lengths[i],i] = nonzero(a[(i*each):(i+1)*each])[0]+1

            else:
                for i in range (0,neuron):
                    tsp[0:lengths[i],i] = nonzero(a[(i*each):(i+1)*each])[0]+1
                for i in range (neuron+1,nn):
                    tsp[0:lengths[i],i-1] = nonzero(a[(i*each):(i+1)*each])[0]+1
        delta = zeros ([p])
        cox = Cox_Method(nn,maxi,target,int_(tsp),delta)
        betahat,betaci = cox.alg2()
        if (neuron == 0):
                betahats[0,1:] = betahat
                betacis[0:2,1:] = betaci.T
        elif (neuron == p):
            betahats [neuron,0:neuron] = betahat
            betacis[nn*3-3:,0:neuron] = betaci.T
        else:
            betahats [neuron,0:neuron] = betahat[0:neuron]
            betahats [neuron,neuron+1:] = betahat [neuron:]
            ind_temp = 3*(neuron+1) - 3
            betacis [ind_temp:ind_temp+2 , 0:neuron] = betaci.T [:,0:neuron]
            betacis [ind_temp:ind_temp+2 , neuron+1:] = betaci.T [:, neuron:]
        print "Neuron " + str(neuron+1) + " out of " + str(nn) + " finished at %s."  %datetime.now()

    for row in betahats:
        row = list(row)

    print (list(betahats))
    print("\n\n\n\n\n")
    print (betacis)

    p = 0
    fro = []
    to = []
    thickness = []
    for i in range (0,nn):
        for j in range (0,nn):
            if (betahats[i,j]>0):
                if ((betacis[p+1,j]> 0.005 and betacis[p,j]>0.005) or (betacis[p+1,j]< -0.005 and betacis[p,j]< -0.005)):
                    fro =  append(fro,j+1)
                    to = append(to,i+1)
                    thickness = append(thickness, betahats[i,j])
        p = p + 3
    to_r = append (append([fro], [to],axis=0), [thickness], axis = 0)
    to_file = to_r.T
    print("\n\n\n\n\n")
    print (to_file)