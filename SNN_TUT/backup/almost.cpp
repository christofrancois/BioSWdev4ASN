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
	string stimtimefile = "";
	string stimfile1 = "/home/user/Documents/stage/test/data/pattern.pat";
	string stimfile2 = "/home/user/Documents/stage/test/data/pattern2.pat";
	string monfile1 = "/home/user/Documents/stage/test/data/monf.pat";
	string monfile2 = "/home/user/Documents/stage/test/data/monf.pat";
	string patternfile = "";

	double time_off = 5;
	double time_on = 50;

	float poisson_rate = 4;
	float weight = 0.125;
	float weight2 = 0.125;
	float sparseness = 0.05;
	float sparseness2 = 0.05;
	int size1 = 28*28; // Adapted to Mnist
	int size2 = size1;

	double scale = 35.0;
	double scale2 = 35.0;
	float gamma = 4.0; //inhib weight scale
	float tau = 10.0; //homeostatic sliding threshold
	float eta = 0.1; //learning rate
	float kappa = 1.0; //target rate
	double background_rate = 10.0;
	double background_rate2 = 10.0;

	/***************INPUT CONSOLE***************/
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("load", po::value<string>(), "input weight matrix")
			("tau", po::value<double>(), "")
			("gamma", po::value<double>(), "")
			("poisson", po::value<double>(), "")
			("spar1", po::value<double>(), "sparseness")
			("spar2", po::value<double>(), "sparseness")
			("bgr1", po::value<double>(), "")
			("bgr2", po::value<double>(), "")
			("kappa", po::value<double>(), "")
			("ton", po::value<double>(), "")
			("toff", po::value<double>(), "")
			("eta", po::value<double>(), "the learning rate")
			("w1", po::value<double>(), "")
			("w2", po::value<double>(), "")
			("sc1", po::value<double>(), "")
			("sc2", po::value<double>(), "")
			("size1", po::value<int>(), "simulation size")
			("size2", po::value<int>(), "simulation size")
			("stimf1", po::value<string>(), "stimulus file")
			("stimf2", po::value<string>(), "stimulus file")
			("monf1", po::value<string>(), "monitor file")
			("monf2", po::value<string>(), "monitor file")
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
		if (vm.count("w1")) {
			weight = vm["w1"].as<double>();
			weight2 = weight;
		}
		if (vm.count("w2")) {
			weight2 = vm["w2"].as<double>();
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
		if (vm.count("kappa")) {
			kappa = vm["kappa"].as<double>();
		}
		if (vm.count("ton")) {
			time_on = vm["ton"].as<double>();
		}
		if (vm.count("toff")) {
			time_off = vm["toff"].as<double>();
		}
		if (vm.count("bgr1")) {
			background_rate = vm["bgr1"].as<double>();
			background_rate2 = background_rate;
		}
		if (vm.count("bgr2")) {
			background_rate2 = vm["bgr2"].as<double>();
		}
		if (vm.count("sc1")) {
			scale = vm["sc1"].as<double>();
			scale2 = scale;
		}
		if (vm.count("sc2")) {
			scale2 = vm["sc2"].as<double>();
		}
		if (vm.count("spar1")) {
			sparseness = vm["spar1"].as<double>();
			sparseness2 = sparseness;
		}
		if (vm.count("spar2")) {
			sparseness2 = vm["spar2"].as<double>();
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
		if (vm.count("stimf1")) {
			stimfile1 = vm["stimf1"].as<string>();
		}
		if (vm.count("stimf2")) {
			stimfile2 = vm["stimf"].as<string>();
		}
		if (vm.count("monf1")) {
			monfile1 = vm["monf1"].as<string>();
		}
		if (vm.count("monf2")) {
			monfile2 = vm["monf2"].as<string>();
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
	IFGroup * n_exc2 = new IFGroup(nb_exc);
	n_exc->get_state_vector("g_ndma")->set_random();

	// Creating inhibitory population
	IFGroup * n_inh = new IFGroup(nb_inh);
	n_inh->set_tau_mem(5e-3);
	IFGroup * n_inh2 = new IFGroup(nb_inh);
	n_inh->set_tau_mem(5e-3);

	// Creating stimulus population
	sprintf(strbuf, "%s/%d.stimtimes", dir.c_str(), sys->mpi_rank());
	stimtimefile = strbuf;

	PoissonGroup * n_stim = new PoissonGroup(nb_stim, poisson_rate);
	/*PoissonGroup * n_stim2 = new PoissonGroup(nb_stim, poisson_rate);
	StimulusGroup * n_stim = new StimulusGroup(nb_stim, stimtimefile);
	n_stim->set_mean_on_period(1.0); // Average length of stim = 1s
	n_stim->set_mean_off_period(5.0); // Average wait before stim = 5s
	n_stim->binary_patterns = true;
	n_stim->scale = scale;
	n_stim->background_rate = background_rate;
	n_stim->background_during_stimulus = true;
*/
	StimulusGroup * n_stim2 = new StimulusGroup(nb_stim, stimtimefile);
	n_stim2->set_mean_on_period(1.0); // Average length of stim = 1s
	n_stim2->set_mean_off_period(5.0); // Average wait before stim = 5s
	n_stim2->binary_patterns = true;
	n_stim2->scale = scale2;
	n_stim2->background_rate = background_rate2;
	n_stim2->background_during_stimulus = true;

	// Creating all connections between the assemblies
	SparseConnection * con_se = new SparseConnection(
		n_stim, n_exc,
		weight, sparseness,
		GLUT
	);
	SparseConnection * con_s2e2 = new SparseConnection(
		n_stim2, n_exc2,
		weight2, sparseness2,
		GLUT
	);
	TripletConnection * con_ee = new TripletConnection(
		n_exc, n_exc,
		weight, sparseness,
		tau, eta, kappa
	);
	con_ee->set_transmitter(GLUT);
	con_ee->stdp_active = false;
	TripletConnection * con_e2e2 = new TripletConnection(
		n_exc2, n_exc2,
		weight2, sparseness2,
		tau, eta, kappa
	);
	con_e2e2->set_transmitter(GLUT);
	con_e2e2->stdp_active = false;
	/*TripletConnection * con_ee2 = new TripletConnection(
		n_exc, n_exc2,
		weight, sparseness * 2,
		tau, eta, kappa
	);
	con_ee2->set_transmitter(GLUT);
	con_ee2->stdp_active = false;
	TripletConnection * con_e2e = new TripletConnection(
		n_exc2, n_exc,
		weight, sparseness,
		tau, eta, kappa
	);
	con_e2e->set_transmitter(GLUT);
	con_e2e->stdp_active = false;
	SparseConnection * con_ei = new SparseConnection(
		n_exc, n_inh,
		weight, sparseness,
		GLUT
	);*/
	SparseConnection * con_e2i2 = new SparseConnection(
		n_exc2, n_inh2,
		weight2, sparseness2,
		GLUT
	);
	SparseConnection * con_ie = new SparseConnection(
		n_inh, n_exc,
		weight * gamma, sparseness,
		GABA
	);
	SparseConnection * con_i2e2 = new SparseConnection(
		n_inh2, n_exc2,
		weight2 * gamma, sparseness2,
		GABA
	);
	SparseConnection * con_ii = new SparseConnection(
		n_inh, n_inh,
		weight * gamma, sparseness,
		GABA
	);
	SparseConnection * con_i2i2 = new SparseConnection(
		n_inh2, n_inh2,
		weight2 * gamma, sparseness2,
		GABA
	);

	// Differents Monitors for the outputs
	sprintf(strbuf, "%s/%d.e.prate", dir.c_str(), sys->mpi_rank());
	PopulationRateMonitor * pmon_e = new PopulationRateMonitor(
		n_exc, string(strbuf)
	);
	sprintf(strbuf, "%s/%d.e2.prate", dir.c_str(), sys->mpi_rank());
	PopulationRateMonitor * pmon_e2 = new PopulationRateMonitor(
		n_exc2, string(strbuf)
	);
	/*sprintf(strbuf, "%s/%d.e.ras", dir.c_str(), sys->mpi_rank());
	SpikeMonitor * smon_e = new SpikeMonitor(
		n_exc, string(strbuf)
	);
	sprintf(strbuf, "%s/%d.e2.ras", dir.c_str(), sys->mpi_rank());
	SpikeMonitor * smon_e2 = new SpikeMonitor(
		n_exc2, string(strbuf)
	);*/

	// Load stim
	/*if(!stimfile1.empty()){
		logger->msg("Setting up stimulus 1 ...",PROGRESS,true);
		n_stim->load_patterns(stimfile1.c_str());
		n_stim->set_next_action_time(25);
	}
	else{
		logger->msg("No stim 1 found");
	}*/
	/*aif(!stimfile2.empty()){
		logger->msg("Setting up stimulus 2 ...",PROGRESS,true);
		//n_stim2->load_patterns(stimfile2.c_str());
		n_stim2->load_patterns(stimfile1.c_str());
		n_stim2->set_next_action_time(25);
	}
	else{
		logger->msg("No stim 2 found");
	}*/


	// Run the simulation for X seconds
	sys->run(time_off);
	con_ee->stdp_active = true;
	//con_e2e->stdp_active = true;
	//con_ee2->stdp_active = true;
	con_e2e2->stdp_active = true;
	sys->run(time_on);

	// Write code above
	auryn_free();
}
