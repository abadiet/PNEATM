#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <PNEATM/genome.hpp>
#include <PNEATM/species.hpp>
#include <PNEATM/Connection/connection.hpp>
#include <PNEATM/Connection/innovation_connection.hpp>
#include <PNEATM/Node/innovation_node.hpp>
#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
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

/**
 * @brief A template class representing a population.
 * @tparam Args Variadic template arguments that contains all the manipulated types.
 */
template <typename... Args>
class Population {
	public:
		/**
		 * @brief Constructor for the Population class.
		 * @param popSize The size of the population.
		 * @param bias_sch The biases scheme (e.g., there is bias_sch[k] biases for type of index k).
		 * @param inputs_sch The inputs scheme (e.g., there is inputs_sch[k] inputs for type of index k).
		 * @param outputs_sch The outputs scheme (e.g., there is outputs_sch[k] outputs for type of index k).
		 * @param hiddens_sch_init The initial hidden nodes scheme (e.g., there is hiddens_sch_init[i][j] hidden nodes of input type of index i and output type of index j).
		 * @param bias_values The initial biases values (e.g., k-th bias will have value bias_values[k]).
		 * @param resetValues The biases reset values (e.g., k-th bias can be resetted to resetValues[k]).
		 * @param activationFns The activation functions (e.g., activationFns[i][j] is a pointer to an activation function that takes an input of type of index i and return a type of index j output).
		 * @param N_ConnInit The initial number of connections.
		 * @param probRecuInit The initial probability of recurrence.
		 * @param weightExtremumInit The initial weight extremum.
		 * @param maxRecuInit The maximum recurrence value.
		 * @param logger A pointer to the logger for logging.
		 * @param dstType The distance function.
		 * @param speciationThreshInit The initial speciation threshold. (default is 20.0)
		 * @param threshGensSinceImproved The maximum number of generations without any improvement. (default is 15)
		 * @param stats_filepath The filepath for statistics. (default is an empty string, which doesn't create any file)
		 */
		Population (unsigned int popSize, std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_values, std::vector<void*> resetValues, std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns, unsigned int N_ConnInit, double probRecuInit, double weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger, distanceFn dstType = CONVENTIONAL, double speciationThreshInit = 20.0, unsigned int threshGensSinceImproved = 15, std::string stats_filepath = "");

		Population (const std::string& filepath, spdlog::logger* logger, std::string stats_filepath = "");

		/**
		 * @brief Destructor for the Population class.
		 */
		~Population ();

		//Population (const std::string filepath) {load(filepath);};

		/**
		 * @brief Get the current generation number.
		 * @return The current generation number.
		 */
		unsigned int getGeneration () {return generation;};

		/**
		 * @brief Get the average fitness.
		 * @return The average fitness.
		 */
		double getAvgFitness () {return avgFitness;};

		/**
		 * @brief Get the adjusted average fitness.
		 * @return The adjusted average fitness.
		 */
		double getAvgFitnessAdjusted () {return avgFitnessAdjusted;};

		/**
		 * @brief Get a reference to the Genome with the specified ID.
		 * @param id The ID of the Genome to retrieve. If set to any negative number, the fitter genome will be returned. (default is -1)
		 * @return A reference to the Genome with the specified ID.
		 */
		Genome<Args...>& getGenome (int id = -1);

		/**
		 * @brief Load the inputs for the entire population.
		 * @tparam T_in The type of input data.
		 * @param inputs A vector containing inputs to be loaded.
		 */
		template <typename T_in>
		void loadInputs(std::vector<T_in> inputs);

		/**
		 * @brief Load a single input for the entire population.
		 * @tparam T_in The type of input data.
		 * @param input The input data to be loaded.
		 * @param input_id The ID of the input to load.
		 */
		template <typename T_in>
		void loadInput(T_in input, unsigned int input_id);

		/**
		 * @brief Load the inputs for a specific genome.
		 * @tparam T_in The type of input data.
		 * @param inputs A vector containing inputs to be loaded.
		 * @param genome_id The ID of the genome for which to load the inputs.
		 */
		template <typename T_in>
		void loadInputs(std::vector<T_in> inputs, unsigned int genome_id);

		/**
		 * @brief Load a single input data for a specific genome.
		 * @tparam T_in The type of input data.
		 * @param input The input data to be loaded.
		 * @param input_id The ID of the input to load.
		 * @param genome_id The ID of the genome for which to load the input.
		 */
		template <typename T_in>
		void loadInput(T_in input, unsigned int input_id, unsigned int genome_id);

		/**
		 * @brief Reset the memory of the entire population.
		 */
		void resetMemory ();

		/**
		 * @brief Reset the memory of a specific genome.
		 * @param genome_id The ID of the genome for which to reset the memory.
		 */
		void resetMemory (unsigned int genome_id);

		/**
		 * @brief Run the network of the entire population.
		 */
		void runNetwork ();

		/**
		 * @brief Run the network of a specific genome.
		 * @param genome_id The ID of the genome for which to run the newtork.
		 */
		void runNetwork (unsigned int genome_id);

		/**
		 * @brief Get the outputs of a specific genome.
		 * @tparam T_out The type of output data.
		 * @param genome_id The ID of the genome for which to get the outputs.
		 * @return A vector containing the outputs of the specified genome.
		 */
		template <typename T_out>
		std::vector<T_out> getOutputs (unsigned int genome_id);

		/**
		 * @brief Get a specific output of a specific genome.
		 * @tparam T_out The type of output data.
		 * @param output_id The ID of the output data.
		 * @param genome_id The ID of the genome for which to get the outputs.
		 * @return A vector containing the specified output of the specified genome.
		 */
		template <typename T_out>
		T_out getOutput (unsigned int output_id, unsigned int genome_id);

		/**
		 * @brief Set the fitness of a specific genome.
		 * @param fitness The fitness to set on.
		 * @param genome_id The ID of the genome for which to set the fitness.
		 */
		void setFitness (double fitness, unsigned int genome_id);

		/**
		 * @brief Assign genomes to species based on their similarity.
		 * @param target The target number of species. (default is 5)
		 * @param maxIterationsReachTarget The maximum number of iterations to reach the target species count. (default is 100)
		 * @param stepThresh The stepsize for adjusting the speciation threshold. (default is 0.3)
		 * @param a Coefficient for computing the excess genes contribution to the distance [for CONVENTIONAL distance only]. (default is 1.0)
		 * @param b Coefficient for computing the disjoint genes contribution to the distance [for CONVENTIONAL distance only]. (default is 1.0)
		 * @param c Coefficient for computing the average weight difference contribution to the distance [for CONVENTIONAL distance only]. (default is 0.4)
		 * @param speciesSizeEvolutionLimit The limit for the evolution of species size. (default is 3.0)
		 */
		void speciate (unsigned int target = 5, unsigned int maxIterationsReachTarget = 100, double stepThresh = 0.3, double a = 1.0, double b = 1.0, double c = 0.4, double speciesSizeEvolutionLimit = 3.0);

		/**
		 * @brief Perform crossover operation to create the new generation.
		 * @param elitism Set to true to keep the fitter genome in the new generation. (default is false)
		 * @param crossover_rate The probability of performing crossover for each new genome. (default is 0.9)
		 */
		void crossover (bool elitism = false, double crossover_rate = 0.9);

		/**
		 * @brief Perform mutation operations over the entire population.
		 * @param params Mutation parameters.
		 */
		void mutate (mutationParams_t params);

		/**
		 * @brief Perform mutation operations for the entire population.
		 * @param paramsMap A function that returns mutation parameters relative to the genome's fitness.
		 */
		void mutate (std::function<mutationParams_t (double)> paramsMap);

		/**
		 * @brief Print information on the population.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (std::string prefix = "");

		/**
		 * @brief Draw a graphical representation of the specified genome's network.
		 * @param genome_id The ID of the genome to draw.
		 * @param font_path The filepath of the font to be used for labels.
		 * @param windowWidth The width of the drawing window. (default is 1300)
		 * @param windowHeight The height of the drawing window. (default is 800)
		 * @param dotsRadius The radius of the dots representing nodes. (default is 6.5f)
		 */
		void drawGenome (unsigned int genome_id, std::string font_path, unsigned int windowWidth = 1300, unsigned int windowHeight = 800, float dotsRadius = 6.5f);

		void save (const std::string& filepath);

		void load (const std::string& filepath);

		void serialize (std::ofstream& outFile);

		void deserialize (std::ifstream& inFile);

	private:
		unsigned int generation;
		double avgFitness;
		double avgFitnessAdjusted;
		unsigned int popSize;
		double speciationThresh;
		unsigned int threshGensSinceImproved;

		// useful parameters to create new genome
		std::vector<size_t> bias_sch;
		std::vector<size_t> inputs_sch;
		std::vector<size_t> outputs_sch;
		std::vector<std::vector<size_t>> hiddens_sch_init;
		std::vector<void*> bias_values;
		std::vector<void*> resetValues;
		unsigned int N_ConnInit;
		double probRecuInit;
		double weightExtremumInit;
		unsigned int maxRecuInit;

		distanceFn dstType;

		int fittergenome_id;
		std::vector<std::unique_ptr<Genome<Args...>>> genomes;
		std::vector<Species<Args...>> species;
		std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
		innovationConn_t conn_innov;
		innovationNode_t node_innov;	// node's innovation id is more like a global id to decerne two different nodes than something to track innovation

		spdlog::logger* logger;
		std::ofstream statsFile;

		std::vector<Connection> GetWeightedCentroid (unsigned int speciesId);
		void UpdateFitnesses (double speciesSizeEvolutionLimit);
		int SelectParent (unsigned int iSpe);

};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename... Args>
Population<Args...>::Population(unsigned int popSize, std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_values, std::vector<void*> resetValues, std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns, unsigned int N_ConnInit, double probRecuInit, double weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger, distanceFn dstType, double speciationThreshInit, unsigned int threshGensSinceImproved, std::string stats_filepath) :
	popSize (popSize),
	speciationThresh (speciationThreshInit),
	threshGensSinceImproved (threshGensSinceImproved),
	bias_sch (bias_sch),
	inputs_sch (inputs_sch),
	outputs_sch (outputs_sch),
	hiddens_sch_init (hiddens_sch_init),
	bias_values (bias_values),
	resetValues (resetValues),
	N_ConnInit (N_ConnInit),
	probRecuInit (probRecuInit),
	weightExtremumInit (weightExtremumInit),
	maxRecuInit (maxRecuInit),
	dstType (dstType),
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
	avgFitness = 0.0;
	avgFitnessAdjusted = 0.0;

	genomes.reserve (popSize);
	for (unsigned int i = 0; i < popSize; i++) {
		genomes.push_back (std::make_unique<Genome<Args...>> (bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_values, resetValues, activationFns, &conn_innov, &node_innov, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger));
	}
}

template <typename... Args>
Population<Args...>::Population (const std::string& filepath, spdlog::logger* logger, std::string stats_filepath) :
	logger (logger)
{
	logger->info ("Population loading");
	if (stats_filepath != "") {
		statsFile.open (stats_filepath);
		statsFile << "Generation,Best Fitness,Average Fitness,Average Fitness (Adjusted),Species0,Species1\n";
	}

	load (filepath);
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
void Population<Args...>::resetMemory () {
	for (size_t i = 0; i < genomes.size (); i++) {
		logger->trace ("Reset genome{}'s memory", i);
		genomes [i]->resetMemory ();
	}
}

template <typename... Args>
void Population<Args...>::resetMemory (unsigned int genome_id) {
	logger->trace ("Reset genome{}'s memory", genome_id);
	genomes [genome_id]->resetMemory ();
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
void Population<Args...>::setFitness (double fitness, unsigned int genome_id) {
	logger->trace ("Setting genome{}'s fitness", genome_id);
	genomes [genome_id]->fitness = fitness;
}

template <typename... Args>
void Population<Args...>::speciate (unsigned int target, unsigned int maxIterationsReachTarget, double stepThresh, double a, double b, double c, double speciesSizeEvolutionLimit) {
	logger->info ("Speciation");

	std::vector<Species<Args...>> tmpspecies;
	unsigned int nbSpeciesAlive = 0;
	unsigned int ite = 0;

	while (ite < maxIterationsReachTarget && nbSpeciesAlive != target) {

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

				size_t itmpspeciesBest;
				double dstBest;
				if (tmpspecies.size () > 0) {	// if there is at least one species

					// we search for the closest species
					itmpspeciesBest = 0;
					dstBest = tmpspecies [itmpspeciesBest].distanceWith (genomes [genome_id], a, b, c);
					double dst;
					for (size_t itmpspecies = 0; itmpspecies < tmpspecies.size (); itmpspecies++) {
						if (!tmpspecies [itmpspecies].isDead && (dst = tmpspecies [itmpspecies].distanceWith (genomes [genome_id], a, b, c)) <= dstBest) {
							// we found a closer one
							itmpspeciesBest = itmpspecies;
							dstBest = dst;
						}
					}

				}
				if (dstBest >= speciationThresh || tmpspecies.size () == 0) {
					// the closest species is too far or there is no species: we create one new
					itmpspeciesBest = tmpspecies.size ();
					tmpspecies.push_back (Species<Args...> ((unsigned int) itmpspeciesBest, genomes [genome_id]->connections, dstType));
				}

				tmpspecies [itmpspeciesBest].members.push_back (genome_id);
				genomes [genome_id]->speciesId = tmpspecies [itmpspeciesBest].id;
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
		if (nbSpeciesAlive < target) {
			speciationThresh -= stepThresh;
		} else {
			if (nbSpeciesAlive > target) {
				speciationThresh += stepThresh;
			}
		}

		ite++;
	}

	// Set tmpspecies as species
	species.clear ();
	species = tmpspecies;

	logger->trace ("speciation result in {0} alive species in {1} iteration(s)", nbSpeciesAlive, ite);
	if ((float) nbSpeciesAlive > (float) target * 1.3f || (float) nbSpeciesAlive < (float) target * 0.7f) {
		logger->warn ("There is a huge difference between target ({0}) and the current number of species ({1})", target, nbSpeciesAlive);
	}

	// update species
	for (size_t iSpe = 0; iSpe < species.size (); iSpe++) {
		if (!species [iSpe].isDead) {	// if the species is still alive, this also ensure that there is at least one member
			//species [iSpe].connections = GetWeightedCentroid ((unsigned int) iSpe);

			size_t leaderId = species [iSpe].members [0];
			for (size_t i = 1; i < species [iSpe].members.size (); i++) {
				if (genomes [species [iSpe].members [i]]->fitness >= genomes [leaderId]->fitness) {
					if (genomes [species [iSpe].members [i]]->fitness > genomes [leaderId]->fitness || Random_Double (0.0, 1.0, true, false) < 0.5) {
						leaderId = species [iSpe].members [i];
					}
				}
			}
			species [iSpe].connections = genomes [leaderId]->connections;	// species leader is the more fit genome
		}
	}

	// update all the fitness as we now know the species
	UpdateFitnesses (speciesSizeEvolutionLimit);
}

template <typename... Args>
std::vector<Connection> Population<Args...>::GetWeightedCentroid (unsigned int speciesId) {
	std::vector<Connection> result;

	double sumFitness = 0.0;

	for (size_t i = 0; i < species [speciesId].members.size (); i++) {	// for each genome in the species
		double fitness = genomes [species [speciesId].members [i]]->fitness;

		for (const Connection& conn : genomes [species [speciesId].members [i]]->connections) {	// for each of its connections
			if (conn.enabled) {	// only pay attention to active ones
				size_t curResConn = 0;
				while (curResConn < result.size () && result [curResConn].innovId != conn.innovId) {	// while we have not found any corresponding connection in result
					curResConn++;
				}
				if (curResConn >= result.size ()) {	// there is no corresponding connections, we add it
					result.push_back (conn);
					result [curResConn].weight = 0.0;	// set its weight as null as all the previous genomes doesn't contains it
				}
				result [curResConn].weight += conn.weight * fitness;	// we add the connection's weight dot the genome's fitness (weighted centroid, check below)
			}
		}

		sumFitness += fitness;
	}

	// we divide each weight by sumFitness to have the average (weighted centroid)
	if (sumFitness > 0.0) {
		for (size_t i = 0; i < result.size (); i++) {
			result [i].weight /= sumFitness;
		}
	} else {
		// null sumFitness
		for (size_t i = 0; i < result.size (); i++) {
			result [i].weight = std::numeric_limits<double>::max ();
		}
	}

	return result;
}

template <typename... Args>
void Population<Args...>::UpdateFitnesses (double speciesSizeEvolutionLimit) {
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
	avgFitness /= (double) popSize;

	// process avgFitnessAdjusted
	for (size_t i = 0; i < species.size (); i++) {
		if (!species [i].isDead) {
			// process species' sumFitness
			species [i].sumFitness = 0;
			for (size_t j = 0; j < species [i].members.size (); j++) {
				species [i].sumFitness += genomes [species [i].members [j]]->fitness;
			}

			// update species' gensSinceImproved
			if (species [i].sumFitness / (double) species[i].members.size () > species [i].avgFitness) {
				// the avgFitness of the species has increased
				species [i].gensSinceImproved  = 0;
			} else {
				species [i].gensSinceImproved ++;
			}

			// process species' avgFitness and avgFitnessAdjusted
			species [i].avgFitness = species [i].sumFitness / (double) species [i].members.size ();
			species [i].avgFitnessAdjusted = species [i].avgFitness / (double) species [i].members.size ();

			avgFitnessAdjusted += species [i].avgFitness;
		}
	}
	avgFitnessAdjusted /= (double) popSize;

	// process offsprings
	for (size_t i = 0; i < species.size (); i ++) {
		if (!species [i].isDead) {
			if (species [i].gensSinceImproved < threshGensSinceImproved) {
				// the species can have offsprings

				double evolutionFactor = species [i].avgFitnessAdjusted / (avgFitnessAdjusted + std::numeric_limits<double>::min ());
				if (evolutionFactor > speciesSizeEvolutionLimit) evolutionFactor = speciesSizeEvolutionLimit;	// we limit the speceis evolution factor: a species's size cannot skyrocket from few genomes

				species [i].allowedOffspring = (int) ((double) species [i].members.size () * evolutionFactor);	// note that (int) 0.9 == 0.0
			} else {
				// the species cannot have offsprings it has not improved for a long time
				species[i].allowedOffspring = 0;
				logger->trace ("species{} has not improved for a long time: it is removed", i);
			}
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
void Population<Args...>::crossover (bool elitism, double crossover_rate) {
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

				// choose pseudo-randomly a first parent
				unsigned int iParent1 = SelectParent (iSpe);

				if (Random_Double (0.0, 1.0, true, false) < crossover_rate && species [iSpe].members.size () > 1) {
					// choose pseudo-randomly a second parent
					unsigned int iParent2 = SelectParent (iSpe);	// TODO might be the same parent as iParent1: is that an issue?

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
								if (Random_Double (0.0, 1.0, true, false) < 0.5) {	// 50 % of chance for each parent, newGenome already have the wheight of MainParent
									newGenomes.back ()->connections [iMainParentConn].weight = genomes [iSecondParent]->connections [iSecondParentConn].weight;
								}
							}
						}
					}
				} else {
					// the genome is kept for the new generation (there is no crossover which emphasize mutation's effect eg exploration)
					logger->trace ("adding genome{} to the new generation", iParent1);
					newGenomes.push_back (genomes [iParent1]->clone ());
				}
			}
		}
	}

	int previousSize = (int) newGenomes.size();
	// add genomes if some are missing
	for (int k = 0; k < (int) popSize - (int) previousSize; k++) {
		newGenomes.push_back (std::make_unique<Genome<Args...>> (bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_values, resetValues, activationFns, &conn_innov, &node_innov, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger));
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

	if (Eq_Double (species [iSpe].sumFitness, 0.0)) {
		// everyone as a null fitness: we return a random genome
		return species [iSpe].members [
			Random_UInt (0, (unsigned int) species [iSpe].members.size () - 1)
		];
	}

	double randThresh = Random_Double (0.0, species [iSpe].sumFitness, true, false);
	double runningSum = 0.0;
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
void Population<Args...>::mutate (std::function<mutationParams_t (double)> paramsMap) {
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
	std::cout << prefix << "   Weight's range at intialization: [" << -1.0 * weightExtremumInit << ", " << weightExtremumInit << "]" << std::endl;
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


template <typename... Args>
void Population<Args...>::serialize (std::ofstream& outFile) {
	Serialize (generation, outFile);
    Serialize (avgFitness, outFile);
    Serialize (avgFitnessAdjusted, outFile);
    Serialize (popSize, outFile);
    Serialize (speciationThresh, outFile);
    Serialize (threshGensSinceImproved, outFile);
    Serialize (bias_sch, outFile);
	Serialize (inputs_sch, outFile);
	Serialize (outputs_sch, outFile);
	Serialize (hiddens_sch_init, outFile);
	Serialize (N_ConnInit, outFile);
    Serialize (probRecuInit, outFile);
    Serialize (weightExtremumInit, outFile);
    Serialize (maxRecuInit, outFile);
    Serialize (dstType, outFile);
    Serialize (fittergenome_id, outFile);
    Serialize (genomes.size (), outFile);
	for (size_t i = 0; i < genomes.size (); i++) {
		genomes [i]->serialize (outFile);
	}
    Serialize (species.size (), outFile);
	for (size_t i = 0; i < species.size (); i++) {
		species [i].serialize (outFile);
	}
    Serialize (activationFns.size (), outFile);
	for (size_t i = 0; i < activationFns.size (); i++) {

    	Serialize (activationFns [i].size (), outFile);
		for (size_t j = 0; j < activationFns [i].size (); j++) {
	
    		Serialize (activationFns [i][j].size (), outFile);
			for (size_t k = 0; k < activationFns [i][j].size (); k++) {
				activationFns [i][j][k]->serialize (outFile);
			}
		}
	}
	conn_innov.serialize (outFile);
	node_innov.serialize (outFile);
}

template <typename... Args>
void Population<Args...>::deserialize (std::ifstream& inFile) {
	Deserialize (generation, inFile);
    Deserialize (avgFitness, inFile);
    Deserialize (avgFitnessAdjusted, inFile);
    Deserialize (popSize, inFile);
    Deserialize (speciationThresh, inFile);
    Deserialize (threshGensSinceImproved, inFile);
    Deserialize (bias_sch, inFile);
	Deserialize (inputs_sch, inFile);
	Deserialize (outputs_sch, inFile);
	Deserialize (hiddens_sch_init, inFile);
	Deserialize (N_ConnInit, inFile);
    Deserialize (probRecuInit, inFile);
    Deserialize (weightExtremumInit, inFile);
    Deserialize (maxRecuInit, inFile);
    Deserialize (dstType, inFile);
    Deserialize (fittergenome_id, inFile);
}

template <typename... Args>
void Population<Args...>::save (const std::string& filepath) {
	std::ofstream outFile(filepath, std::ios::binary);
	if (!outFile) {
		logger->error ("Cannot open file {} for writing.", filepath);
		return;
	}

	serialize (outFile);

	outFile.close ();
}

template <typename... Args>
void Population<Args...>::load (const std::string& filepath) {
	std::ifstream inFile(filepath, std::ios::binary);
	if (!inFile) {
		logger->error ("Cannot open file {} for reading.", filepath);
		return;
	}

	deserialize (inFile);

	inFile.close ();
}

#endif	// POPULATION_HPP
