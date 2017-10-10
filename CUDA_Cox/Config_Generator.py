from numpy import *
import random as ran
def config_generator (number_of_neurons,max_conn_per_neuron,save_path):
    '''
    An ELIF model neuron can be simulated using software from the following web-site: http://www.tech.plymouth.ac.uk/infovis. To run\
    the simulation, the parameters of ELIF neurons and their coupling should be specified. This function generates the \
    configuration files needed for that software.

    :param number_of_neurons: Number of neurons in the network.
    :param max_conn_per_neuron: maximum number of connection that each neuron might have with other neurons.
    :param save_path: path to save the generated configuration files.
    '''
    with open(save_path+'np.np', 'w+') as f:

        for idx in range(1,number_of_neurons+1):
            max_thr = 45 + (random.rand(1) * random.randint(-2,3,size=1)+random.rand(1))
            dec_thr = 2 + random.rand(1)
            thr_inf = 13.5 + (random.rand(1) * random.randint(-1,1,size=1)+random.rand(1))
            amp_ns = 5 + (random.rand(1) * random.randint(-1,1,size=1)+random.rand(1))
            dec_ns = 9.9 + random.rand(1) * 0.1
            init_pot = -28 - random.rand(1)
            dec_mpt = 20 + (random.rand(1) * random.randint(-1,1,size=1)+random.rand(1))
            ext_inp = 0.2 + random.rand(1) * 0.5
            ref = random.randint(3,8,size=1)
            type = 0
            f.write("neuron%d\n %f %f %f %f %f %f %f %f %d %d\n"%(idx,max_thr,dec_thr,thr_inf,amp_ns,dec_ns,init_pot,dec_mpt,ext_inp,ref,type))

    with open(save_path+'cp.cp', 'w+') as f:
        for idx in range(1, number_of_neurons+1):
            conn_num = array(random.randint(1,max_conn_per_neuron+1))
            sample = arange(1,number_of_neurons+1)
            sample = delete(sample,idx-1)
            cons_idx = array(ran.sample(sample, int(conn_num)))
            strength = array(map ( lambda x : float(11 + (random.rand(1) * random.randint(-2,4,size=1)+random.rand(1))) , ones([conn_num])))
            decays = array(map ( lambda x : float(3 + random.rand(1) ) , ones([conn_num])))
            lag = array(map ( lambda x : int(11 +  random.randint(-3,4,size=1)) , ones([conn_num])))
            f.write("neuron%d\n%d\n\t\t\t%s\n\t\t\t\t\t%s\n\t\t\t\t\t\t\t%s\n\t\t\t\t\t\t\t\t\t\t%s\n"%(idx,conn_num,array_str(cons_idx)[1:-1],array_str(strength)[1:-1],array_str(decays)[1:-1],array_str(lag)[1:-1]))