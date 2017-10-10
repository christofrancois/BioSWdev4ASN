__author__ = 'V_AD'


from Cox_Method import Cox_Method
from numpy import *
from matplotlib.pyplot import *
import pickle

def main_CRCNS (data_path,recording_length,window_length,overlap,minimum_spike):
    '''
    This is the main function that applies the cox method on the pickle data saved by CRCNS_to_COX. This function plots the \
    adjacency matrix of connection weights in a window-based manner.

    :param data_path: The path to the pickle data file.
    :param recording_length: Total duration of recording of the dataset. This can be found in the metadata-tables of the \
    experiments too, e.g. in case of ec012ec.187 experiment.
    :param window_length: The length of the window on which the cox method is going to apply on.
    :param overlap: the overlap of the window with its previous window.

    Main internal variables:

    * nn: Total number of neurons in the network.
    * p: Number of reference neurons in the network.
    * win_num: Number of windows on data based on the length of data, length of window and overlap size.
    * final_adjacency: The result of the function in form of adjaceency matrix.
    * l_band: Lower band of the window.
    * u_band: Higher band of the window.
    * betahats: Matrix of betahat values in the network.
    * betacis: Matrix of confidence interval of betahat values in the network.
    * Xs: Reference neuron indices (x-axis) for drawing the final connectivity matrix
    * Ys: Target neuron indices (y-axis) for drawing the final connectivity matrix
    * Ss: Strength of the connection between each target neuron and corresponding reference neurons
    '''
    with open (data_path + '/crcns_data.pickle','rb') as handle:
        spikes = pickle.load(handle)
    nn = len(spikes) # Total number of neurons in the network
    p = nn-1 # Number of reference neurons in the network
    radius_to_add =2
    win_num = int(float(recording_length)/(window_length-overlap)) # Number of windows on data based on the length of data, length of window and overlap size.
    print "number of windows:%d"%win_num
    final_adjacency = zeros ([nn,nn]) # The result of the function in form of adjaceency matrix
    for win_n in range (win_num):
        l_band = win_n * window_length if win_n == 0 else (win_n * window_length)-overlap # Lower band of the window
        u_band = l_band + window_length # Higher band of the window.
        temp_spikes = {}
        for qp in spikes:
            temp_spikes[qp] = array([tt for tt in spikes[qp] if (tt <u_band and tt>l_band)])
        temp_lengths = array([len(temp_spikes[q]) for q in temp_spikes])
        non_small_indices = array(where(temp_lengths>minimum_spike)[0])
        selected_spikes = {}
        for non_small in non_small_indices :
            selected_spikes[non_small] = temp_spikes[non_small]
        nn_for_cox = len(non_small_indices)
        betahats = zeros((nn_for_cox,nn_for_cox)) # Matrix of betahat values in the network.
        betacis = zeros ((nn_for_cox*3-1 , nn_for_cox)) # Matrix of confidence interval of betahat values in the network.
        for neuron in range ( 0, nn_for_cox) :
            target_b = selected_spikes[non_small_indices[neuron]]
            if (neuron == 0):
                print ("Windows No. %d Length of each train in average: " %win_n)
                print(temp_lengths[non_small_indices].astype(int))
                print(average(non_small_indices))
            maxi_b = 0
            for i,q in list (enumerate (non_small_indices)):
                if i!= neuron and temp_lengths[q] > maxi_b :
                    maxi_b = temp_lengths[q]
            ref_b = zeros((maxi_b,nn_for_cox-1))-1
            idx = 0
            for q in range ( 0, nn_for_cox):
                if non_small_indices[q] != non_small_indices[neuron] :
                    ref_b[0:temp_lengths[non_small_indices[q]],idx] = selected_spikes[non_small_indices[q]]
                    idx+=1
            tsp_b = ref_b
            delta = zeros ([nn_for_cox-1])
            cox = Cox_Method(nn_for_cox,maxi_b , target_b, tsp_b.astype(int), delta)
            betahat, betaci = cox.alg1()
            if (neuron == 0):
                    betahats[0,1:] = betahat
                    betacis[0:2,1:] = betaci.T
            elif (neuron == p):
                betahats [neuron,0:neuron] = betahat
                betacis [nn*3-3:,0:neuron] = betaci.T
            else:
                betahats [neuron,0:neuron] = betahat[0:neuron]
                betahats [neuron,neuron+1:] = betahat [neuron:]
                ind_temp = 3*(neuron+1) - 3
                betacis [ind_temp:ind_temp+2 , 0:neuron] = betaci.T [:,0:neuron]
                betacis [ind_temp:ind_temp+2 , neuron+1:] = betaci.T [:, neuron:]
        fro = array ([])
        to = array ([])
        p_temp = 0
        for i in range (0,nn_for_cox):
            for j in range (0,nn_for_cox):
                if (betahats[i,j]>0):
                    if ((betacis[p_temp+1,j]> 0 and betacis[p_temp,j]>0) or (betacis[p_temp+1,j]< 0 and betacis[p_temp,j]< 0)):
                        final_adjacency[non_small_indices[j], non_small_indices[i]] += radius_to_add
                        fro = append(fro,non_small_indices[j])
                        to = append(to,non_small_indices[i])
            p_temp = p_temp + 3
        Xs = arange(1,nn+1) # Reference neuron indices (x-axis) for drawing the final connectivity matrix
        Ys = arange(1,nn+1) # Target neuron indices (y-axis) for drawing the final connectivity matrix
        Ss = ones([nn])*10 #  Strength of the connection between each target neuron and corresponding reference neurons
        for y in range(nn) :
            for x in range(nn) :
                if final_adjacency[x,y] != 0 :
                    Xs = append(Xs,x)
                    Ys= append(Ys,y)
                    Ss = append(Ss,final_adjacency[x,y] )
    with open ('PATH/Xs','wb') as h1:
        pickle.dump(Xs, h1)
    with open ('PATH/Ys','wb') as h2:
        pickle.dump(Ys, h2)
    with open ('PATH/Ss','wb') as h3:
        pickle.dump(Ss, h3)

    fig = figure ( )
    ax = fig.gca()
    ax.set_xticks(arange(-0.5,35.5,1))
    ax.set_yticks(arange(-0.5,35.5,1))
    scatter (Xs,Ys,s=Ss)
    axis('equal')
    ax.set_xlim([0,35])
    ax.set_ylim([0,35])
    grid()
    show()

if __name__ == '__main__' :
    recording_length = 1096400
    window_length = 7000
    minimum_spike_rate = 200
    overlap = 100
    main_CRCNS('./test', recording_length, window_length, overlap,minimum_spike_rate)