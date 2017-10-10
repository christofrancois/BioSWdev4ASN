#include "auryn.h"
using namespace auryn;

int main(int ac, char* av[])
{
  auryn_init( ac, av );

  // the simulation code will go here

  char strbuf [255];

  // Creating a balanced network
  int nb_exc = 20000;
  int nb_inh = nb_exc / 4;
  IFGroup * n_exc = new IFGroup(nb_exc);
  n_exc->get_state_vector("g_nmda")->set_random();
  IFGroup * n_inh = new IFGroup(nb_inh);
  n_inh->set_tau_mem(5e-3);

  int nb_stim = nb_exc / 4;
  float poisson_rate = 2.0;
  PoissonGroup * n_stim = new PoissonGroup(nb_stim, poisson_rate);

  // Connecting the network
  float weight = 0.2;
  float sparseness = 0.05;
  //  input
  SparseConnection * con_se = new SparseConnection(
    n_stim, n_exc, weight, sparseness, GLUT
  );
  //  recurrent connections
  float gamma = 4.0; //stronger connections for inhibitory neurons
  SparseConnection * con_ee = new SparseConnection(
    n_exc, n_exc, weight, sparseness, GLUT
  );
  SparseConnection * con_ei = new SparseConnection(
    n_exc, n_inh, weight, sparseness, GLUT
  );
  SparseConnection * con_ii = new SparseConnection(
    n_inh, n_inh, gamma * weight, sparseness, GABA
  );
  SparseConnection * con_ie = new SparseConnection(
    n_inh, n_exc, gamma * weight, sparseness, GABA
  );

  // Setting up Monitor
  sprintf(strbuf, "./res2/secondtest/e.ras");
  SpikeMonitor * s_mon = new SpikeMonitor(n_exc, string(strbuf));
  sprintf(strbuf, "./res2/secondtest/e.mem");
  VoltageMonitor * v_mon = new VoltageMonitor(n_exc, 0, string(strbuf));

  // Run the simulation for 10 seconds
  sys->run(10);

  auryn_free();
}
