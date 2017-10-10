#include "auryn.h"
using namespace auryn;
namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac, char* av[])
{
	/***************PROGRAM VARIABLES***************/
	char strbuf [255];
	string msg;

	string results = "/home/user/Documents/stage/test/results";
	string stimfiles = "/home/user/Documents/stage/test/data/stimfiles";
	string stimtimes = "/home/user/Documents/stage/test/data/stimtimes";
	string trig = "/home/user/Documents/stage/test/data/trig";
	string loadfile = "";

	int nb_exc = 4096*2;
	int nb_inh = nb_exc / 4;
	int nb_stim = nb_exc / 2;

	int nb_exc2 = nb_exc / 2;
	int nb_inh2 = nb_exc2 / 4;
	int nb_stim2 = nb_exc2 / 2;

	float poisson_rate = 1.0;
	float tau_hom = 5;
	float eta_rel = 0.1;
	float kappa = 1.0;

	float weight = 0.2;
	float sparseness = 0.05;
	float gamma = 4.0;

	bool save = false;
	bool load = false;
	bool stim = false;
	bool lab = false;
	bool init = false;
	double ton = 0;
	int rank = 0;
	bool ras = false;

	/***************INPUT CONSOLE***************/
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("load", "input weight matrix")
			("init", "init")
			("rank", po::value<int>(), "rank")
			("save", "save network state at end of sim")
			("stim", "allow stim")
			("lab", "allow labels")
			("ras", "allow ras")
			("ontime", po::value<double>(), "simulation on time")
		;
		po::variables_map vm;
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			std::cout << desc << "\n";
			return 1;
		}
		if (vm.count("load")) {
			load = true;
    	}
		if (vm.count("init")) {
			init = true;
    	}
		if (vm.count("save")) {
			save = true;
    	}
		if (vm.count("stim")) {
			stim = true;
    	}
		if (vm.count("lab")) {
			lab = true;
    	}
		if (vm.count("ontime")) {
			ton = vm["ontime"].as<double>();
    	}
		if (vm.count("rank")) {
			rank = vm["rank"].as<int>();
		}
		if (vm.count("ras")) {
			ras = true;
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


	// Create the groups of neurons
	// Excitatory
	IFGroup * n_exc = new IFGroup(nb_exc);
	n_exc->set_name("exc neurons 1");
	n_exc->get_state_vector("g_ndma")->set_random();
	// Inhibitory
	IFGroup * n_inh = new IFGroup(nb_inh);
	n_inh->set_name("inh neurons 1");
	n_inh->set_tau_mem(5e-3);
	// Stimulus Group
	sprintf(strbuf, "%s/original.stimtimes", stimtimes.c_str());
	string st1 = strbuf;
	sprintf(strbuf, "%s/mnist.pat", stimfiles.c_str());
	string sf1 = strbuf;
	StimulusGroup * n_stim = new StimulusGroup(
		nb_stim, sf1.c_str(), st1.c_str(), STIMFILE	);
	n_stim->binary_patterns = true;
	n_stim->scale = 55;
	n_stim->background_rate = 2;
	n_stim->background_during_stimulus = true;

	//Second layer
	IFGroup * n_exc2 = new IFGroup(nb_exc2);
	n_exc2->set_name("exc neurons 2");
	n_exc2->get_state_vector("g_ndma")->set_random();
/*	IFGroup * n_inh2 = new IFGroup(nb_inh2);
	n_inh2->set_name("inh neurons 2");
	n_inh2->set_tau_mem(5e-3);*/

	sprintf(strbuf, "%s/original2.stimtimes", stimtimes.c_str());
	string st2 = strbuf;
	sprintf(strbuf, "%s/labels.pat", stimfiles.c_str());
	string sf2 = strbuf;
	StimulusGroup * n_stim2 = new StimulusGroup(
		nb_stim2, sf2.c_str(), st2.c_str(), STIMFILE	);
	n_stim2->binary_patterns = true;
	n_stim2->scale = 35;
	n_stim2->background_rate = 2;
	n_stim2->background_during_stimulus = true;


	// Setting up the connections


	// Layer 1 :
	SparseConnection * con_se = new SparseConnection(
		n_stim, n_exc,
		weight, sparseness,
		GLUT	);
	TripletConnection * con_ee = new TripletConnection(
		n_exc, n_exc,
		weight, sparseness,
		tau_hom, eta_rel,	kappa	);
	con_ee->set_transmitter(GLUT);
	con_ee->stdp_active = false;
	SparseConnection * con_ei = new SparseConnection(
		n_exc, n_inh,
		weight, sparseness,
		GLUT	);
	SparseConnection * con_ie = new SparseConnection(
		n_inh, n_exc,
		gamma * weight, sparseness,
		GABA	);
	SparseConnection * con_ii = new SparseConnection(
		n_inh, n_inh,
		gamma * weight, sparseness,
		GABA	);

	// Layer 2 :
	SparseConnection * con_s2e2 = new SparseConnection(
		n_stim2, n_exc2,
		weight, sparseness * 1,
		GLUT	);
/*	TripletConnection * con_e2e2 = new TripletConnection(
		n_exc2, n_exc2,
		weight, sparseness * 1,
		tau_hom, eta_rel,	kappa	);
	con_e2e2->set_transmitter(GLUT);
	con_e2e2->stdp_active = false;
	SparseConnection * con_e2i2 = new SparseConnection(
		n_exc2, n_inh2,
		weight, sparseness * 1,
		GLUT	);
	SparseConnection * con_i2e2 = new SparseConnection(
		n_inh2, n_exc2,
		gamma * weight, sparseness * 1,
		GABA	);
	SparseConnection * con_i2i2 = new SparseConnection(
		n_inh2, n_inh2,
		gamma * weight, sparseness * 1,
		GABA	);*/

	// Connection between two layers
	TripletConnection * con_ee2 = new TripletConnection(
		n_exc, n_exc2,
		weight / 2, sparseness / 2,
		tau_hom, eta_rel,	kappa	);


	// Monitors

	if(ras){
		// spike raster
		sprintf(strbuf, "%s/%d->%d.e.spk", results.c_str(), rank, sys->mpi_rank());
		BinarySpikeMonitor * smon_e = new BinarySpikeMonitor(
		n_exc2, string(strbuf)	);
	}
	// layer 2 pattern activity
	sprintf(strbuf, "%s/%d->%d.e2.pact", results.c_str(), rank, sys->mpi_rank());
  string pact = strbuf;
  sprintf(strbuf, "%s/e2.pat", trig.c_str());
  string patt = strbuf;
	PatternMonitor * patmon = new PatternMonitor(
		n_exc2,
    pact, patt);

	if (init) {
		// initial burn before activating plasticity
		logger->progress("init");
		n_stim->set_next_action_time(20);
		n_stim2->set_next_action_time(20);
		sys->run(10);
	}

	if(load) {
		// loading the weight matrix
		logger->progress("load");
		sprintf(strbuf, "%s/%d.ee2.wmat", results.c_str(), sys->mpi_rank());
		/*con_ee2->load_from_file(strbuf);
		sprintf(strbuf, "%s/%d.e2e2.wmat", results.c_str(), sys->mpi_rank());
		con_e2e2->load_from_file(strbuf);*/
		sprintf(strbuf, "%s/%d.ee.wmat", results.c_str(), sys->mpi_rank());
		con_ee->load_from_file(strbuf);
	}

	if (!stim) {
		// disable MNIST input
		logger->progress("stim disabled");
		n_stim->set_next_action_time(ton);
	}
	if (!lab) {
		// disable Labels input
		logger->progress("lab disabled");
		n_stim2->set_next_action_time(ton);
	}
	// activate plasticity
	con_ee->stdp_active = true;
	//con_e2e2->stdp_active = true;
	// runs the simulation
	sys->run(ton);

	if (save) {
		// save the weight matrix
		sys->set_output_dir(results);
		sys->save_network_state("save");
		sprintf(strbuf, "%s/%d.ee.wmat", results.c_str(), sys->mpi_rank() );
		con_ee->write_to_file(strbuf);
		/*sprintf(strbuf, "%s/%d.e2e2.wmat", results.c_str(), sys->mpi_rank() );
		con_e2e2->write_to_file(strbuf);*/
		sprintf(strbuf, "%s/%d.ee2.wmat", results.c_str(), sys->mpi_rank() );
		con_ee2->write_to_file(strbuf);
	}

	// Write code above
	auryn_free();
}
