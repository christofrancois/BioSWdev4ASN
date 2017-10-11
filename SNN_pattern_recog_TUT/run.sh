<<COMMENT1
	$> make && ./run.sh

	runs all phases of the simulation in order
	1) training with MNIST and LABELS stimulus 
		and 10s of initial burn before turning the plasticity on
	2) runs with only labels stimulus and generate a spike file 
		to analyze with find_patterns.sh
		it will find the 10 subpopulations fo neurons responding
		to each classes
	3) runs the test analyze the file "3->0.e2.pact" which contains
		the activity 
COMMENT1

#clean the results directory
rm results/*

#generate the stimulus files
cd scripts/generate
./generate_all.sh
cd ../..

#runs the simulations
#	1°
mpirun -n 4 ./sim --rank 1 --init --stim --lab --on 3600 --save
#	2°
mpirun -n 4 ./sim --rank 2 --ras --lab --on 1000 --load
# find the 10 sub populations
cd scripts/generate
./find_patterns.sh
cd ../..
#	3°
mpirun -n 4 ./sim --rank 3 --stim --on 1000 --load