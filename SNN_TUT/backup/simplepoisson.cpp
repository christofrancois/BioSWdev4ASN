#include "auryn.h"
using namespace auryn;
namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac, char* av[])
{
	/***************PROGRAM VARIABLES***************/
	char strbuf [255];
	string msg;

	string dir = "/home/user/Documents/stage/test/results/";
	string data = "/home/user/Documents/stage/test/data/";
	string infilename = "";
	string simtime = "";
	string stimfile1 = "";
	string stimfile2 = "";
	string monfile1 = "";
	string monfile2 = "";

	double time_off = 5;
	double time_on = 10;

	float poisson_rate = 4;
	float weight = 0.125;
	float sparseness = 0.05;
	int size1 = 28*28; // Adapted to Mnist
	int size2 = size1 / 4;

	float gamma = 4.0; //inhib weight scale
	float tau = 10.0; //homeostatic sliding threshold
	float eta = 0.1; //learning tate
	float kappa = 1.0; //target rate

	/***************INPUT CONSOLE***************/
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("load", po::value<string>(), "input weight matrix")
			("tau", po::value<double>(), "")
			("gamma", po::value<double>(), "")
			("poisson", po::value<double>(), "")
			("sparseness", po::value<double>(), "sparseness")
			("ton", po::value<double>(), "")
			("toff", po::value<double>(), "")
			("eta", po::value<double>(), "the learning rate")
			("weight", po::value<double>(), "")
			("size1", po::value<int>(), "simulation size")
			("size2", po::value<int>(), "simulation size")
			("stimfile1", po::value<string>(), "stimulus file")
			("stimfile2", po::value<string>(), "stimulus file")
			("monfile", po::value<string>(), "monitor file")
		;

		po::variables_map vm;
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return 1;
		}
		if (vm.count("load")) {
			infilename = vm["load"].as<string>();
		}
		if (vm.count("weight")) {
			weight = vm["weight"].as<double>();
		}
		if (vm.count("poisson")) {
			poisson_rate = vm["poisson"].as<double>();
		}
		if (vm.count("tau")) {
			tau = vm["tau"].as<double>();
		}
		if (vm.count("gamma")) {
			gamma = vm["gamma"].as<double>();
		}
		if (vm.count("ton")) {
			time_on = vm["ton"].as<double>();
		}
		if (vm.count("toff")) {
			time_off = vm["toff"].as<double>();
		}
		if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
		}
		if (vm.count("sparseness")) {
			sparseness = vm["sparseness"].as<double>();
		}
		if (vm.count("eta")) {
			eta = vm["eta"].as<double>();
		}
		if (vm.count("size1")) {
			size1 = vm["size1"].as<int>();
		}
		if (vm.count("size2")) {
			size2 = vm["size2"].as<int>();
		}
		if (vm.count("stimfile1")) {
			stimfile1 = vm["stimfile1"].as<string>();
			monfile1 = stimfile1;
		}
		if (vm.count("stimfile2")) {
			stimfile2 = vm["stimfile2"].as<string>();
			monfile2 = stimfile2;
		}
		if (vm.count("monfile")) {
			monfile1 = vm["monfile"].as<string>();
		}
	}
	catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
		return 1;
	}
	catch(...) {
		std::cerr << "Exception of unknown type!\n";
	}

	/***************MAIN PROGRAM***************/

	auryn_init(ac, av);
	// Write code below

	int nb_exc = size1;
	int nb_inh = nb_exc/4;
	int nb_stim = nb_exc;

	// Creating excitatory population
	IFGroup * n_exc = new IFGroup(nb_exc);
	n_exc->get_state_vector("g_ndma")->set_random();

	// Creating inhibitory population
	IFGroup * n_inh = new IFGroup(nb_inh);
	n_inh->set_tau_mem(5e-3);

	// Creating stimulus population
	PoissonGroup * n_stim = new PoissonGroup(nb_stim, poisson_rate);

	// Creating all connections between the assemblies
	SparseConnection * con_se = new SparseConnection(
		n_stim, n_exc,
		weight, sparseness,
		GLUT
	);
	TripletConnection * con_ee = new TripletConnection(
		n_exc, n_exc,
		weight, sparseness,
		tau, eta, kappa
	);
	con_ee->set_transmitter(GLUT);
	con_ee->stdp_active = false;
	SparseConnection * con_ei = new SparseConnection(
		n_exc, n_inh,
		weight, sparseness,
		GLUT
	);
	SparseConnection * con_ie = new SparseConnection(
		n_inh, n_exc,
		weight * gamma, sparseness,
		GABA
	);
	SparseConnection * con_ii = new SparseConnection(
		n_inh, n_inh,
		weight * gamma, sparseness,
		GABA
	);

	// Differents Monitors for the outputs
	sprintf(strbuf, "%s/%d.e.prate", dir.c_str(), sys->mpi_rank());
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor(
		n_exc, string(strbuf)
	);
	sprintf(strbuf, "%s/%d.i.prate", dir.c_str(), sys->mpi_rank());
	PopulationRateMonitor * pmon_i = new PopulationRateMonitor(
		n_inh, string(strbuf)
	);
	sprintf(strbuf, "%s/%d.e.ras", dir.c_str(), sys->mpi_rank());
	SpikeMonitor * smon_e = new SpikeMonitor(
		n_exc, string(strbuf)
	);

	// Run the simulation for X seconds
	sys->run(time_off);
	con_ee->stdp_active = true;
	sys->run(time_on);

	// Abort if the network crashes
	RateChecker * chk = new RateChecker( n_exc , 0.1 , 1000. , 100e-3);

	// Write code above
	auryn_free();
}
