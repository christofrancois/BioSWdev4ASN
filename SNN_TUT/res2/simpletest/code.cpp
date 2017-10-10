#include "auryn.h"
using namespace auryn;

int main(int ac, char* av[])
{
  auryn_init( ac, av );

  // the simulation code will go here
  int nb_stim = 100;
  float poisson_rate = 3;
  float weight = 0.2;
  char strbuf [255];

  // Populations of neurons :
  // Stimulus neurons
  PoissonGroup * n_stim = new PoissonGroup(nb_stim, poisson_rate);
  // Receiver neuron
  IFGroup * n_exc1 = new IFGroup(1);

  // Monitors :
  //  spike rasterplot of stimulus neurons
  sprintf(strbuf, "./res2/simpletest/s.ras");
  SpikeMonitor * sp_mon = new SpikeMonitor(n_stim, string(strbuf));
  //  membrane voltage
  sprintf(strbuf, "./res2/simpletest/e1.mem");
  VoltageMonitor * v_mon1 = new VoltageMonitor(n_exc1, 0, string(strbuf));

  // Run 3 seconds without connections
  sys->run(3);

  // Set the connections from Stim to Exc
  AllToAllConnection * con_se = new AllToAllConnection(n_stim, n_exc1, weight);


  // Run the simulation for 10 seconds
  sys->run(10);

  auryn_free();
}
