__author__ = 'V_AD'

from numpy import *
import pickle
def CRCNS_to_Cox(data_path, filename_prefix, electrodes, path_to_save):
    '''
    This file convert the data from CRCNS datasets to a format usable for Cox method. \
    The main file used from the datasets are ".clu" and ".res" files.

    :param data_path: Path of the dataset downloaded from www.CRCNS.org
    :param data_path: The prefix of the files entitled after the name of the experiment, e.g. ec012ec.187
    :param electrodes: Number of electrodes in the measurement.
    :param path_to_save: Path for saving the result (it will be imported in main file of cox method)
    '''

    spikes_dic = {}
    counter = 0
    spikes_dic['electrodes'] = {}
    for elec in range (1,electrodes+1):
        str_to_run = '''with open (data_path+"\\\\"+ filename_prefix+".clu.%d") as clust:
    with open (data_path + "\\\\" +  filename_prefix +".res.%d") as spikes:
        idx = array([int(v1) for v1 in clust])
        neuron_added = int(idx[0])
        idx= idx[1:]
        time =  array([int(float(v1)*0.05) for v1 in spikes])
        spikes_dic['electrodes'][elec] = zeros([neuron_added])
        for neuron in range (neuron_added):
            temp_spikes = time [where (idx == neuron)]
            spikes_dic['electrodes'][elec][neuron] = int(len(temp_spikes))
            spikes_dic['electrodes'][elec] =spikes_dic['electrodes'][elec].astype(int)
            spikes_dic[counter] = temp_spikes
            counter+=1
                   ''' % (elec, elec)

        exec str_to_run
    with open (path_to_save+ '/crcns_data.pickle','wb') as handle:
        pickle.dump(spikes_dic, handle)


CRCNS_to_Cox('E:\CRCNS\ec012ec.11\ec012ec.187','ec012ec.187',4,"d:/test")