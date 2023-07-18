#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <PNEATM/genome.hpp>
#include <PNEATM/species.hpp>
#include <PNEATM/Connection/connection.hpp>
#include <PNEATM/Connection/innovation_connection.hpp>
#include <PNEATM/Node/innovation_node.hpp>
#include <PNEATM/utils.hpp>
#include <fstream>
#include <iostream>
#include <cstring>
#include <limits>
#include <vector>
#include <spdlog/spdlog.h>
#include <memory>
#include <functional>

/* HEADER */

namespace pneatm {

template <typename... Args>
class Population {
	public:
		Population (unsigned int popSize, std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_init, std::vector<void*> resetValues, std::vector<std::vector<std::vector<void*>>> activationFns, unsigned int N_ConnInit, float probRecuInit, float weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger, float speciationThreshInit = 10.0f, unsigned int threshGensSinceImproved = 15, std::string stats_filepath = "");
		~Population ();
		//Population (const std::string filepath) {load(filepath);};

		unsigned int getGeneration () {return generation;};
		float getAvgFitness () {return avgFitness;};
		float getAvgFitnessAdjusted () {return avgFitnessAdjusted;};
		Genome<Args...>& getGenome (int id = -1);

		template <typename T_in>
		void loadInputs (std::vector<T_in> inputs);
		template <typename T_in>
		void loadInput (T_in input, unsigned int input_id);
		template <typename T_in>
		void loadInputs (std::vector<T_in> inputs, unsigned int genome_id);
		template <typename T_in>
		void loadInput (T_in input, unsigned int input_id, unsigned int genome_id);

		void runNetwork ();
		void runNetwork (unsigned int genome_id);

		template <typename T_out>
		std::vector<T_out> getOutputs (unsigned int genome_id);
		template <typename T_out>
		T_out getOutput (unsigned int output_id, unsigned int genome_id);

		void setFitness (float fitness, unsigned int genome_id);
		void speciate (unsigned int target = 5, unsigned int targetThresh = 0, unsigned int maxIterationsReachTarget = 100, float stepThresh = 0.3f, float a = 1.0f, float b = 1.0f, float c = 0.4f, float speciesSizeEvolutionLimit = 3.0f);
		void crossover (bool elitism = false);
		void mutate (mutationParams_t params);
		void mutate (std::function<mutationParams_t (float)> paramsMap);

		void print (std::string prefix = "");
		void drawGenome (unsigned int genome_id, std::string font_path, unsigned int windowWidth = 1300, unsigned int windowHeight = 800, float dotsRadius = 6.5f);

		/*void save (const std::string filepath = "./neat_backup.txt");
		void load (const std::string filepath = "./neat_backup.txt");*/

	private:
		unsigned int generation;
		float avgFitness;
		float avgFitnessAdjusted;
		unsigned int popSize;
		float speciationThresh;
		unsigned int threshGensSinceImproved;

		// useful parameters to create new genome
		std::vector<size_t> bias_sch;
		std::vector<size_t> inputs_sch;
		std::vector<size_t> outputs_sch;
		std::vector<std::vector<size_t>> hiddens_sch_init;
		std::vector<void*> bias_init;
		std::vector<void*> resetValues;
		unsigned int N_ConnInit;
		float probRecuInit;
		float weightExtremumInit;
		unsigned int maxRecuInit;

		int fittergenome_id;
		std::vector<std::unique_ptr<Genome<Args...>>> genomes;
		std::vector<Species<Args...>> species;
		std::vector<std::vector<std::vector<void*>>> activationFns;
		innovationConn_t conn_innov;
		innovationNode_t node_innov;	// innovation id is more like a global id to decerne two different nodes than something to track innovation

		spdlog::logger* logger;
		std::ofstream statsFile;

		std::vector<Connection> GetWeightedCentroid (unsigned int speciesId);
		void UpdateFitnesses (float speciesSizeEvolutionLimit);
		int SelectParent (unsigned int iSpe);
};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename... Args>
Population<Args...>::Population(unsigned int popSize, std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_init, std::vector<void*> resetValues, std::vector<std::vector<std::vector<void*>>> activationFns, unsigned int N_ConnInit, float probRecuInit, float weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger, float speciationThreshInit, unsigned int threshGensSinceImproved, std::string stats_filepath) :
	popSize (popSize),
	speciationThresh (speciationThreshInit),
	threshGensSinceImproved (threshGensSinceImproved),
	bias_sch (bias_sch),
	inputs_sch (inputs_sch),
	outputs_sch (outputs_sch),
	hiddens_sch_init (hiddens_sch_init),
	bias_init (bias_init),
	resetValues (resetValues),
	N_ConnInit (N_ConnInit),
	probRecuInit (probRecuInit),
	weightExtremumInit (weightExtremumInit),
	maxRecuInit (maxRecuInit),
	activationFns (activationFns),
	logger (logger)
{
	logger->info ("Population initialization");
	if (stats_filepath != "") {
		statsFile.open (stats_filepath);
		statsFile << "Generation,Best Fitness,Average Fitness,Average Fitness (Adjusted),Species0,Species1\n";
	} 

	generation = 0;
	fittergenome_id = -1;

	genomes.reserve (popSize);
	for (unsigned int i = 0; i < popSize; i++) {
		genomes.push_back (std::make_unique<Genome<Args...>> (bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, &conn_innov, &node_innov, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger));
	}
}

template <typename... Args>
Population<Args...>::~Population () {
	logger->info ("Population destruction");
	if (statsFile.is_open ()) statsFile.close ();
}

template <typename... Args>
Genome<Args...>& Population<Args...>::getGenome (int id) {
	if (id < 0 || id >= (int) popSize) {
		if (fittergenome_id < 0) {
			// fitter genome cannot be found
			logger->warn ("Calling Population<Args...>::getGenome cannot determine which is the more fit genome: in order to know it, call Population<Args...>::speciate first. Returning the first genome.");
			return *genomes [0];
		}
		return *genomes [fittergenome_id];
	}
	return *genomes [id];
}

template <typename... Args>
template <typename T_in>
void Population<Args...>::loadInputs(std::vector<T_in> inputs) {
	for (int i = 0; i < popSize; i++) {
		logger->trace ("Load genome{}'s inputs", i);
		genomes [i]->template loadInputs<T_in> (inputs);
	}
}

template <typename... Args>
template <typename T_in>
void Population<Args...>::loadInputs(std::vector<T_in> inputs, unsigned int genome_id) {
	logger->trace ("Load genome{}'s inputs", genome_id);
	genomes [genome_id]->template loadInputs<T_in> (inputs);
}

template <typename... Args>
template <typename T_in>
void Population<Args...>::loadInput(T_in input, unsigned int input_id) {
	for (unsigned int i = 0; i < popSize; i++) {
		logger->trace ("Load genome{0}'s input{1}", i, input_id);
		genomes [i]->template loadInput<T_in> (input, input_id);
	}
}

template <typename... Args>
template <typename T_in>
void Population<Args...>::loadInput(T_in input, unsigned int input_id, unsigned int genome_id) {
	logger->trace ("Load genome{0}'s input{1}", genome_id, input_id);
	genomes [genome_id]->template loadInput<T_in> (input, input_id);
}

template <typename... Args>
void Population<Args...>::runNetwork () {
	for (unsigned int i = 0; i < popSize; i++) {
		logger->trace ("Run genome{0}'s network", i);
		genomes [i]->runNetwork ();
	}
}

template <typename... Args>
void Population<Args...>::runNetwork(unsigned int genome_id) {
	logger->trace ("Run genome{0}'s network", genome_id);
	genomes [genome_id]->runNetwork ();
}

template <typename... Args>
template <typename T_out>
std::vector<T_out> Population<Args...>::getOutputs (unsigned int genome_id) {
	logger->trace ("Get genome{}'s outputs", genome_id);
	return genomes [genome_id]->template getOutputs<T_out> ();
}

template <typename... Args>
template <typename T_out>
T_out Population<Args...>::getOutput (unsigned int output_id, unsigned int genome_id) {
	logger->trace ("Get genome{0}'s output{1}", genome_id, output_id);
	return genomes [genome_id]->template getOutput<T_out> (output_id);
}

template <typename... Args>
void Population<Args...>::setFitness (float fitness, unsigned int genome_id) {
	logger->trace ("setting genome{}'s fitness", genome_id);
	genomes [genome_id]->fitness = fitness;
}

template <typename... Args>
void Population<Args...>::speciate (unsigned int target, unsigned int targetThresh, unsigned int maxIterationsReachTarget, float stepThresh, float a, float b, float c, float speciesSizeEvolutionLimit) {
	logger->info ("Speciation");

	// species randomization: we do that here to avoid the first species to become too large.
	// Actually, as we are using a sequential process to assign a given to genome to a species, the first one may become to large
	std::vector<Species<Args...>> newSpecies;
	while (species.size () > 0) {
		unsigned int r = Random_UInt (0, (unsigned int) species.size () - 1);	// randomly select a species
		newSpecies.push_back (species [r]);										// add it to the new ones
		species.erase (species.begin () + r);									// remove it from the previous ones
	}
	species.clear ();
	species = newSpecies;

	std::vector<Species<Args...>> tmpspecies;
	unsigned int nbSpeciesAlive = 0;
	unsigned int ite = 0;

	while (
		ite < maxIterationsReachTarget
		&& ((int) nbSpeciesAlive < (int) target - (int) targetThresh || nbSpeciesAlive > target + targetThresh)
	) {

		// init tmpspecies
		tmpspecies.clear ();
		tmpspecies = species;
		nbSpeciesAlive = 0;

		// reset species
		for (unsigned int i = 0; i < popSize; i++) {
			genomes [i]->speciesId = -1;
		}

		// reset tmpspecies
		for (size_t iSpe = 0; iSpe < tmpspecies.size (); iSpe++) {
			if (!tmpspecies [iSpe].isDead) {	// if the species is still alive
				tmpspecies [iSpe].members.clear ();
			}
		}

		// process the other genomes
		for (unsigned int genome_id = 0; genome_id < popSize; genome_id++) {
			if (genomes [genome_id]->speciesId == -1) {	// if the genome not already belong to a species
				unsigned int itmpspecies = 0;
				while (
					itmpspecies < (unsigned int) tmpspecies.size ()
					&& (
						tmpspecies [itmpspecies].isDead
						|| !(tmpspecies [itmpspecies].distanceWith (genomes [genome_id], a, b, c) < speciationThresh)	// comparison leader vs genome_id
						)
					)
				{
					itmpspecies++;	// the genome cannot belong to this species, let's check the next one
				}
				if (itmpspecies == (unsigned int) tmpspecies.size ()) {
					// no species found for the current genome, we have to create one new
					tmpspecies.push_back (Species<Args...> (itmpspecies, genomes [genome_id]->connections));
				}

				tmpspecies [itmpspecies].members.push_back (genome_id);
				genomes [genome_id]->speciesId = tmpspecies [itmpspecies].id;
			}
		}

		// check how many species are still alive
		for (size_t iSpe = 0; iSpe < tmpspecies.size (); iSpe++) {
			if (tmpspecies [iSpe].members.size () == 0) {
				// the species has no member, the species is dead
				tmpspecies [iSpe].isDead = true;
			}
			if (!tmpspecies [iSpe].isDead) {	// if the species is still alive
				nbSpeciesAlive ++;
			}
		}

		// update speciationThresh
		if ((int) nbSpeciesAlive < (int) target - (int) targetThresh) {
			speciationThresh -= stepThresh;
		} else {
			if (nbSpeciesAlive > target + targetThresh) {
				speciationThresh += stepThresh;
			}
		}

		ite++;
	}

	// Set tmpspecies as species while sorting species by their ids
	species.clear ();
	while (species.size () < tmpspecies.size ()) {
		size_t i = 0;
		while (tmpspecies [i].id != species.size ()) {
			i++;
		}
		species.push_back (tmpspecies [i]);
	}

	logger->trace ("speciation result in {0} alive species in {1} iteration(s)", nbSpeciesAlive, ite);

	// update species
	for (size_t iSpe = 0; iSpe < species.size (); iSpe++) {
		if (!species [iSpe].isDead) {	// if the species is still alive, this also ensure that there is at least one member 
			species [iSpe].connections = GetWeightedCentroid ((unsigned int) iSpe);
		}
	}

	// update all the fitness as we now know the species
	UpdateFitnesses (speciesSizeEvolutionLimit);
}

template <typename... Args>
std::vector<Connection> Population<Args...>::GetWeightedCentroid (unsigned int speciesId) {
	std::vector<Connection> result;

	float sumFitness = 0.0f;

	for (size_t i = 0; i < species [speciesId].members.size (); i++) {	// for each genome in the species
		float fitness = genomes [species [speciesId].members [i]]->fitness;

		for (const Connection& conn : genomes [species [speciesId].members [i]]->connections) {	// for each of its connections
			size_t curResConn = 0;
			while (curResConn < result.size () && result [curResConn].innovId != conn.innovId) {	// while we have not found any corresponding connection in result
				curResConn++;
			}
			if (curResConn >= result.size ()) {	// there is no corresponding connections, we add it
				result.push_back (conn);
				result [curResConn].weight = 0.0f;	// set its weight as null as all the previous genomes doesn't contains it
			}
			result [curResConn].weight += conn.weight * fitness;	// we add the connection's weight dot the genome's fitness (weighted centroid, check below)
		}

		sumFitness += fitness;
	}

	// we divide each weight by sumFitness to have the average (weighted centroid)
	if (sumFitness > 0.0f) {
		for (size_t i = 0; i < result.size (); i++) {
			result [i].weight /= sumFitness;
		}
	} else {
		// null sumFitness
		for (size_t i = 0; i < result.size (); i++) {
			result [i].weight = std::numeric_limits<float>::max ();
		}
	}

	return result;
}

template <typename... Args>
void Population<Args...>::UpdateFitnesses (float speciesSizeEvolutionLimit) {
	fittergenome_id = 0;
	avgFitness = 0;
	avgFitnessAdjusted = 0;

	// process avgFitness and found fittergenome_id
	for (unsigned int i = 0; i < popSize; i++) {
		avgFitness += genomes [i]->fitness;

		if (genomes [i]->fitness > genomes [fittergenome_id]->fitness) {
			fittergenome_id = i;
		}
	}
	avgFitness /= (float) popSize;

	// process avgFitnessAdjusted
	for (size_t i = 0; i < species.size (); i++) {
		if (!species [i].isDead) {
			// process species' sumFitness
			species [i].sumFitness = 0;
			for (size_t j = 0; j < species [i].members.size (); j++) {
				species [i].sumFitness += genomes [species [i].members [j]]->fitness;
			}

			// update species' gensSinceImproved
			if (species [i].sumFitness / (float) species[i].members.size () > species [i].avgFitness) {
				// the avgFitness of the species has increased
				species [i].gensSinceImproved  = 0;
			} else {
				species [i].gensSinceImproved ++;
			}

			// process species' avgFitness and avgFitnessAdjusted
			species [i].avgFitness = species [i].sumFitness / (float) species [i].members.size ();
			species [i].avgFitnessAdjusted = species [i].avgFitness / (float) species [i].members.size ();

			avgFitnessAdjusted += species [i].avgFitness;
		}
	}
	avgFitnessAdjusted /= (float) popSize;

	// process offsprings
	for (size_t i = 0; i < species.size (); i ++) {
		if (!species [i].isDead) {
			if (species [i].gensSinceImproved < threshGensSinceImproved) {
				// the species can have offsprings

				float evolutionFactor = species [i].avgFitnessAdjusted / (avgFitnessAdjusted + std::numeric_limits<float>::min ());
				if (evolutionFactor > speciesSizeEvolutionLimit) evolutionFactor = speciesSizeEvolutionLimit;	// we limit the speceis evolution factor: a species's size cannot skyrocket from few genomes

				species [i].allowedOffspring = (int) ((float) species [i].members.size () * evolutionFactor);	// note that (int) 0.9f == 0.0f
			} else {
				// the species cannot have offsprings it has not improved for a long time
				species[i].allowedOffspring = 0;
				logger->trace ("species{} has not improved for a long time: it is removed", i);
			}

std::cout << species [i].id << "   " << species [i].avgFitnessAdjusted << "   " << species [i].members.size () << "   " << species [i].allowedOffspring << std::endl;
		}
	}

	// add satistics to the file
	if (statsFile.is_open ()) {
		statsFile << generation << "," << genomes [fittergenome_id]->fitness << "," << avgFitness << "," << avgFitnessAdjusted << ",";
		for (size_t i = 0; i < species.size () - 1; i ++) {
			statsFile << species [i].members.size () << ",";
		}
		statsFile << species.back ().members.size () << "\n";
	}
}

template <typename... Args>
void Population<Args...>::crossover (bool elitism) {
	logger->info ("Crossover");
	std::vector<std::unique_ptr<Genome<Args...>>> newGenomes;
	newGenomes.reserve (popSize);

	if (elitism) {	// elitism mode on = we conserve during generations the more fit genome
		logger->trace ("elitism is on: adding the more fit genome to the new generation");
		newGenomes.push_back (genomes [fittergenome_id]->clone ());
	}

	for (unsigned int iSpe = 0; iSpe < (unsigned int) species.size (); iSpe ++) {
		if (!species [iSpe].isDead) {
			for (int k = 0; k < species [iSpe].allowedOffspring; k++) {
				// choose pseudo-randomly two parents. Don't care if they're identical as the child will be mutated...
				unsigned int iParent1 = SelectParent (iSpe);
				unsigned int iParent2 = SelectParent (iSpe);

				// clone the more fit
				unsigned int iMainParent;
				unsigned int iSecondParent;
				if (genomes [iParent1]->fitness > genomes [iParent2]->fitness) {
					iMainParent = iParent1;
					iSecondParent = iParent2;
				} else {
					iMainParent = iParent2;
					iSecondParent = iParent1;
				}

				logger->trace ("adding child from the parents genome{0} and genome{1} to the new generation", iMainParent, iSecondParent);

				newGenomes.push_back (genomes [iMainParent]->clone ());

				// connections shared by both of the parents must be randomly wheighted
				for (size_t iMainParentConn = 0; iMainParentConn < genomes [iMainParent]->connections.size (); iMainParentConn ++) {
					for (size_t iSecondParentConn = 0; iSecondParentConn < genomes [iSecondParent]->connections.size (); iSecondParentConn ++) {
						if (genomes [iMainParent]->connections [iMainParentConn].innovId == genomes [iSecondParent]->connections [iSecondParentConn].innovId) {
							if (Random_Float (0.0f, 1.0f, true, false) < 0.5f) {	// 50 % of chance for each parent, newGenome already have the wheight of MainParent
								newGenomes.back ()->connections [iMainParentConn].weight = genomes [iSecondParent]->connections [iSecondParentConn].weight;
							}
						}
					}
				}
			}
		}
	}

	int previousSize = (int) newGenomes.size();
	// add genomes if some are missing
	for (int k = 0; k < (int) popSize - (int) previousSize; k++) {
		newGenomes.push_back (std::make_unique<Genome<Args...>> (bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, &conn_innov, &node_innov, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger));
	}
	// or remove some genomes if there is too many genomes
	for (int k = 0; k < (int) previousSize - (int) popSize; k++) {
		newGenomes.pop_back ();
	}

	// replace the current genomes by the new ones
	logger->trace ("replacing the genomes");
	genomes.clear ();
	genomes = std::move (newGenomes);

	// reset species members
	for (size_t i = 0; i < species.size(); i++) {
		species [i].members.clear ();
		species [i].isDead = true;
	}
	for (unsigned int i = 0; i < popSize; i++) {
		if (genomes [i]->speciesId > -1) {
			species [genomes [i]->speciesId].members.push_back (i);
			species [genomes [i]->speciesId].isDead = false;	// empty species will stay to isDead = true
		}
	}

	fittergenome_id = -1;	// avoid a missuse of fittergenome_id

	generation ++;
}

template <typename... Args>
int Population<Args...>::SelectParent (unsigned int iSpe) {
	/* Chooses player from the population to return randomly(considering fitness).
	This works by randomly choosing a value between 0 and the sum of all the fitnesses then go through all the genomes
	and add their fitness to a running sum and if that sum is greater than the random value generated,
	that genome is chosen since players with a higher fitness function add more to the running sum then they have a higher chance of being chosen */

	if (Eq_Float (species [iSpe].sumFitness, 0.0f)) {
		// everyone as a null fitness: we return a random genome
		return species [iSpe].members [
			Random_UInt (0, (unsigned int) species [iSpe].members.size () - 1)
		];
	}

	float randThresh = Random_Float (0.0f, species [iSpe].sumFitness, true, false);
	float runningSum = 0.0f;
	for (size_t i = 0; i < species [iSpe].members.size (); i++) {
		runningSum += genomes [species [iSpe].members [i]]->fitness;
		if (runningSum > randThresh) {
			return species [iSpe].members [i];
		}
	}
	return -1;	// impossible
}

template <typename... Args>
void Population<Args...>::mutate (mutationParams_t params) {
	logger->info ("Mutations");
	for (unsigned int i = 0; i < popSize; i++) {
		logger->trace ("Mutation of genome{}", i);
		genomes [i]->mutate (&conn_innov, &node_innov, params);
	}
}

template <typename... Args>
void Population<Args...>::mutate (std::function<mutationParams_t (float)> paramsMap) {
	logger->info ("Mutations");
	for (unsigned int i = 0; i < popSize; i++) {
		logger->trace ("Mutation of genome{}", i);
		genomes [i]->mutate (&conn_innov, &node_innov, paramsMap (genomes [i]->getFitness ()));
	}
}

template <typename... Args>
void Population<Args...>::print (std::string prefix) {
	std::cout << prefix << "Generation Number: " << generation << std::endl;
	std::cout << prefix << "Population Size: " << popSize << std::endl;
	std::cout << prefix << "Current Average Fitness: " << avgFitness << std::endl;
	std::cout << prefix << "Current Average Fitness Adjusted: " << avgFitnessAdjusted << std::endl;
	std::cout << prefix << "Current Speciation Threshold: " << speciationThresh << std::endl;
	std::cout << prefix << "Species die if they does not improve in " << threshGensSinceImproved << " generations" << std::endl;
	std::cout << prefix << "When creating a new Genome: " << std::endl;
	std::cout << prefix << "   Bias Nodes Initialisation [TypeID (Number of Bias Node)]: ";
	for (size_t i = 0; i < bias_sch.size (); i++) {
		std::cout << i << " (" << bias_sch [i] << "), ";
	}
	std::cout << std::endl;
	std::cout << prefix << "   Input Nodes Initialisation [TypeID (Number of Input Node)]: ";
	for (size_t i = 0; i < inputs_sch.size (); i++) {
		std::cout << i << " (" << inputs_sch [i] << "), ";
	}
	std::cout << std::endl;
	std::cout << prefix << "   Output Nodes Initialisation [TypeID (Number of Output Node)]: ";
	for (size_t i = 0; i < outputs_sch.size (); i++) {
		std::cout << i << " (" << outputs_sch [i] << "), ";
	}
	std::cout << std::endl;
	std::cout << prefix << "   Hidden Nodes Initialisation [Input TypeID to Output TypeID2 (Number of Hidden Node)]: ";
	for (size_t i = 0; i < hiddens_sch_init.size (); i++) {
		for (size_t j = 0; j < hiddens_sch_init [i].size (); j++) {
			std::cout << i  << " to " << j << " (" << hiddens_sch_init [i][j] << "), ";
		}
	}
	std::cout << std::endl;
	std::cout << prefix << "   Number of connections at initialization: " << N_ConnInit << std::endl;
	std::cout << prefix << "   Probability of adding recurrency: " << probRecuInit << std::endl;
	std::cout << prefix << "   Maximum recurrency at initialization: " << maxRecuInit << std::endl;
	std::cout << prefix << "   Weight's range at intialization: [" << -1.0f * weightExtremumInit << ", " << weightExtremumInit << "]" << std::endl;
	std::cout << prefix << "Current More Fit Genome ID: " << fittergenome_id << std::endl;
	std::cout << prefix << "Number of Activation Functions [Input TypeID to Output TypeID (Number of functions)]: ";
	for (size_t i = 0; i < activationFns.size (); i++) {
		for (size_t j = 0; j < activationFns [i].size (); j++) {
			std::cout << i << " to " << j << " (" << activationFns [i][j].size () << "), ";
		}
	}
	std::cout << std::endl;
	std::cout << prefix << "Connections Innovations:" << std::endl;
	conn_innov.print (prefix + "   ");
	std::cout << prefix << "Nodes Innovations:" << std::endl;
	node_innov.print (prefix + "   ");
	std::cout << prefix << "Genomes: " << std::endl;
	for (size_t i = 0; i < genomes.size (); i++) {
		std::cout << prefix << " * Genome " << i << std::endl;
		genomes [i]->print (prefix + "   ");
	}
	std::cout << prefix << "Species: " << std::endl;
	for (Species<Args...> spe : species) {
		spe.print (prefix + "   ");
	}
}

template <typename... Args>
void Population<Args...>::drawGenome (unsigned int genome_id, std::string font_path, unsigned int windowWidth, unsigned int windowHeight, float dotsRadius) {
	logger->info ("Drawing genome{}'s network", genome_id);
	genomes [genome_id]->draw (font_path, windowWidth, windowHeight, dotsRadius);
}
/*
template <typename... Args>
void Population<Args...>::save(const std::string filepath){
	std::ofstream fileobj(filepath);

	if (fileobj.is_open()){
		for (int k = 0; k < (int) innovIds.size(); k++){
			for (int j = 0; j < (int) innovIds[k].size(); j++){
				fileobj << innovIds[k][j] << ",";
			}
			fileobj << ";";
		}
		fileobj << "\n";

		fileobj << lastInnovId << "\n";
		fileobj << popSize << "\n";
		fileobj << speciationThresh << "\n";
		fileobj << threshGensSinceImproved << "\n";
		fileobj << nbInput << "\n";
		fileobj << nbOutput << "\n";
		fileobj << nbHiddenInit << "\n";
		fileobj << probConnInit << "\n";
		fileobj << areRecurrentConnectionsAllowed << "\n";
		fileobj << weightExtremumInit << "\n";
		fileobj << generation << "\n";
		fileobj << avgFitness << "\n";
		fileobj << avgFitnessAdjusted << "\n";
		fileobj << fittergenome_id << "\n";

		for (int k = 0; k < (int) genomes.size(); k++){
			fileobj << genomes[k].fitness << "\n";
			fileobj << genomes[k].speciesId << "\n";


			for (int j = 0; j < (int) genomes[k].nodes.size(); j++){
				fileobj << genomes[k].nodes[j].id << ",";
				fileobj << genomes[k].nodes[j].layer << ",";
				fileobj << genomes[k].nodes[j].sumInput << ",";
				fileobj << genomes[k].nodes[j].sumOutput << ",";
			}
			fileobj << "\n";

			for (int j = 0; j < (int) genomes[k].connections.size(); j++){
				fileobj << genomes[k].connections[j].innovId << ",";
				fileobj << genomes[k].connections[j].inNodeId << ",";
				fileobj << genomes[k].connections[j].outNodeId << ",";
				fileobj << genomes[k].connections[j].weight << ",";
				fileobj << genomes[k].connections[j].enabled << ",";
				fileobj << genomes[k].connections[j].isRecurrent << ",";
			}
			fileobj << "\n";
		}

		fileobj.close();
	}
}

template <typename... Args>
void Population<Args...>::load(const std::string filepath){
	std::ifstream fileobj(filepath);

	if (fileobj.is_open()){

		std::string line;
		size_t pos = 0;

		if (getline(fileobj, line)){
			innovIds.clear();
			size_t pos_sep = line.find(';');
			while (pos_sep != std::string::npos) {
				innovIds.push_back({});
				std::string sub_line = line.substr(0, pos_sep - 1);
				pos = sub_line.find(',');
				while (pos != std::string::npos) {
					innovIds.back().push_back(stoi(sub_line.substr(0, pos)));
					sub_line = sub_line.substr(pos + 1);
					pos = sub_line.find(',');
				}
				line = line.substr(pos_sep + 1);
				pos_sep = line.find(';');
			}
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}

		if (getline(fileobj, line)){
			lastInnovId = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			popSize = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			speciationThresh = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			threshGensSinceImproved = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			nbInput = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			nbOutput = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			nbHiddenInit = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			probConnInit = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			areRecurrentConnectionsAllowed = (line == "1");
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			weightExtremumInit = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			generation = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			avgFitness = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			avgFitnessAdjusted = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			fittergenome_id = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		genomes.clear();
		while (getline(fileobj, line)){
			genomes.push_back(Genome(nbInput, nbOutput, nbHiddenInit, probConnInit, &innovIds, &lastInnovId, weightExtremumInit));

			genomes.back().fitness = stof(line);
			if (getline(fileobj, line)){
				genomes.back().speciesId = stoi(line);
			} else {
				std::cout << "Error while loading model" << std::endl;
				throw 0;
			}
			if (getline(fileobj, line)){
				genomes.back().nodes.clear();
				pos = line.find(',');
				while (pos != std::string::npos) {
					genomes.back().nodes.push_back(Node());

					genomes.back().nodes.back().id = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().nodes.back().layer = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().nodes.back().sumInput = (float) stod(line.substr(0, pos));	// stdof's range is not the float's one... weird, anyway it works with stod of course

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().nodes.back().sumOutput = (float) stod(line.substr(0, pos));	// stdof's range is not the float's one... weird, anyway it works with stod of course

					line = line.substr(pos + 1);
					pos = line.find(',');
				}
			} else {
				std::cout << "Error while loading model" << std::endl;
				throw 0;
			}
			if (getline(fileobj, line)){
				genomes.back().connections.clear();
				pos = line.find(',');
				while (pos != std::string::npos) {
					genomes.back().connections.push_back(Connection());

					genomes.back().connections.back().innovId = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().inNodeId = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().outNodeId = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().weight = stof(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().enabled = (line.substr(0, pos) == "1");

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().isRecurrent = (line.substr(0, pos) == "1");

					line = line.substr(pos + 1);
					pos = line.find(',');
				}
			} else {
				std::cout << "Error while loading model" << std::endl;
				throw 0;
			}
		}

		fileobj.close();
	} else {
		std::cout << "Error while loading model" << std::endl;
		throw 0;
	}
}
*/

#endif	// POPULATION_HPP
