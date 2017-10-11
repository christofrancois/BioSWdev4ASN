__author__ = 'V_AD'


# This code is an example of stopping/resuming simulation. At first part of simulation (line 88), the two neuron groups are connected with a low synaptic and input
# weight, resulting in one spike in 500us. At this phase, the results are plotted (line 90 to 98). Closing the result window will lead to the 2nd
# phase of simulation (not from the beginning) in which the weight is increased (line 100) and simulation is continued for another 500us (line 102).

from brian2 import *
from Syntax_Generator import *
from close_to_array import *

NN = 493
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
El = -74*mV
taue = 5*ms
taui = 10*ms
tau_stdp = 20*ms
F = 20*Hz

C = 100*pF
k = 3*pA/mV/mV
vr = -60*mV
vt = -50*mV
a = 0.01*kHz
b = 5*pA/mV
c = -60*mV
d = 400*pA
vpeak = 50*mV
tau_tempfix = 1*ms
memc= 200.0*pfarad
epsilon = 0.2
er = -90 * mV
external = 2000 * pA
eqs_neurons = '''
dv/dt=(k*(v-vr)*(v-vt)-u)/memc + (-ge*v - gi*(v-er)+ external)/memc : volt
du/dt=a*(b*(v-vr)-u) : amp
dge/dt = -ge / taue : siemens
dgi/dt = -gi/taui : siemens
'''

reset = '''
v=c
u+=d
'''

N0 = NeuronGroup(NN, eqs_neurons, threshold='v>vpeak', reset=reset)
Pe = N0 [:347]
Pi = N0 [347:]
pre_ex_all = 'ge +=0.3*nS'
pre_in_in = 'gi +=0.3*nS'
pre_in_ex = '''A_pre += 1.
                            w = clip(w+(A_post-alpha)*eta, 0, gmax)
                           gi += w*nS'''
post_in_ex = '''
                   A_post += 1.
                    w = clip(w+A_pre*eta, 0, gmax)
                   '''
connect=  'rand()<epsilon'
S00 = Synapses (Pe, Pe, pre = pre_ex_all )
S01 = Synapses (Pe, Pi, pre = pre_ex_all )
S11 = Synapses (Pi, Pi, pre = pre_in_in )


eqs_stdp_inhib = '''w : 1
                dA_pre/dt = -A_pre / tau_stdp : 1 (event-driven)
                dA_post/dt = -A_post / tau_stdp : 1 (event-driven)'''
alpha = 1*Hz*tau_stdp*2
gmax = 100
S10 = Synapses (Pi,Pe, model= eqs_stdp_inhib,
                   pre = pre_in_ex ,
                   post = post_in_ex
                   )
######## old results
# syn1, syn2,syn3,syn4 = syn_generator ()
# exec syn1
# exec syn2
# exec syn3
# exec syn4
#####################
syn, con = syn_generator()
closest_indices = close_to_electrode ()
closest_indices = closest_indices.astype(int)
for syns2 in con :
    exec syns2
#     p = random.binomial (1,0.2,1)
#     if p :
    # exec syns2
S10.w =  1e-2
# S00.connect('rand()<epsilon')
# S01.connect('rand()<epsilon')
# S10.connect('rand()<epsilon')
# S11.connect('rand()<epsilon')


s_mon = SpikeMonitor(Pe)

print "starting simulation"

sim_time = 30000*ms
eta = 0
run(1000*ms)
print "'before' is finished"
eta = 0.1
print"starting the second part"
run(sim_time-1000*ms, report='text')

i,t = s_mon.it
pickle.dump( array(i), open( "C:/Users/vafaa/Desktop/results/i.p", "wb" ) )
pickle.dump( array(t), open( "C:/Users/vafaa/Desktop/results/t.p", "wb" ) )

# subplot  (211)
figure()
plot(t/ms , i , '.k', ms= 0.65)
title ("Before")
xlabel("")
# yticks([])
xlim(0*1e3,1*1e3)

# subplot(212)
figure()
plot(t/ms , i , '.k' , ms= 0.65)
xlabel ("time(ms)")
# yticks([])
title ("After ")
xlim((sim_time - 5*second)/ms , sim_time/ms)
# plot(con_e.w / gmax, '.k')
show()
