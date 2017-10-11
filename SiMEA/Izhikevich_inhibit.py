from brian2 import *

NN = 1000
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
El = -74*mV
taue = 5*ms
taui = 10*ms
tau_stdp = 20*ms
F = 20*Hz
# gmax = 10
# dApre = .01
# dApost = -dApre * taupre / taupost * 1.05
# dApost *= gmax
# dApre *= gmax

# Izhikevich Parameters
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


# input = PoissonGroup(1000, rates=F)
N0 = NeuronGroup(NN, eqs_neurons, threshold='v>vpeak', reset=reset)
Pe = N0 [:700]
Pi = N0 [701:]
con_e = Synapses (Pe, N0, pre = 'ge +=0.3*nS' )
con_ii = Synapses (Pi, Pi, pre = 'gi +=0.3*nS')
# con_in = Synapses (input, N0, pre = 'ge +=0.3*nS', connect = 'rand()<epsilon' )
con_e.connect ('rand()<epsilon')
con_ii.connect ('rand()<epsilon')

eqs_stdp_inhib = '''w : 1
                dA_pre/dt = -A_pre / tau_stdp : 1 (event-driven)
                dA_post/dt = -A_post / tau_stdp : 1 (event-driven)'''
alpha = 1*Hz*tau_stdp*2
gmax = 100
con_ie = Synapses (Pi,Pe, model= eqs_stdp_inhib,
                   pre = '''A_pre += 1.
                            w = clip(w+(A_post-alpha)*eta, 0, gmax)
                           gi += w*nS''' ,
                   post = '''
                   A_post += 1.
                    w = clip(w+A_pre*eta, 0, gmax)
                   '''
                   )
con_ie.w =  1e-2
con_ie.connect('rand()<epsilon')



s_mon = SpikeMonitor(N0)


sim_time = 3000*ms
eta = 0
run(1000*ms)
eta = 0.2
run(sim_time-1000*ms, report='text')

i,t = s_mon.it

subplot  (211)
plot(t/ms , i , '.k', ms= 0.25)
title ("Before")
xlabel("")
# yticks([])
xlim(0*1e3,1*1e3)

subplot(212)
plot(t/ms , i , '.k' , ms= 0.25)
xlabel ("time(ms)")
# yticks([])
title ("After ")
xlim((sim_time -1*second)/ms , sim_time/ms)
# plot(con_e.w / gmax, '.k')
show()

