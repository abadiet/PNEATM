#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <PNEATM/genome.hpp>
#include <PNEATM/species.hpp>
#include <PNEATM/Connection/connection.hpp>
#include <PNEATM/Connection/innovation_connection.hpp>
#include <PNEATM/Node/innovation_node.hpp>
#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
#include <PNEATM/Node/Activation_Function/create_activation_function.hpp>
#include <PNEATM/thread_pool.hpp>
#include <PNEATM/utils.hpp>
#include <fstream>
#include <iostream>
#include <cstring>
#include <limits>
#include <vector>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <memory>
#include <functional>
#include <thread>

/* HEADER */

namespace pneatm {

/**
 * @brief A template class representing a population.
 * @tparam Types Variadic template arguments that contains all the manipulated types.
 */
template <typename... Types>
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
		 * @param inputsActivationFns The activation functions of the bias & inputs nodes. The first functions are dedicated to the bias nodes and the other ones to the inputs ones.
		 * @param outputsActivationFns The activation functions of the outputs nodes.
		 * @param N_ConnInit The initial number of connections.
		 * @param probRecuInit The initial probability of recurrence.
		 * @param weightExtremumInit The initial weight extremum.
		 * @param maxRecuInit The maximum recurrence value.
		 * @param logger A pointer to the logger for logging.
		 * @param dstType The distance function.
		 * @param speciationThreshInit The initial speciation threshold. (default is 20.0)
		 * @param threshGensSinceImproved The maximum number of generations without any improvement. (default is 15)
		 * @param stats_filename The filename for statistics. (default is an empty string, which doesn't create any file)
		 */
		Population (unsigned int popSize, const std::vector<size_t>& bias_sch, const std::vector<size_t>& inputs_sch, const std::vector<size_t>& outputs_sch, const std::vector<std::vector<size_t>>& hiddens_sch_init, const std::vector<void*>& bias_values, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, const std::vector<ActivationFnBase*> inputsActivationFns, const std::vector<ActivationFnBase*> outputsActivationFns, unsigned int N_ConnInit, double probRecuInit, double weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger, distanceFn dstType = CONVENTIONAL, double speciationThreshInit = 20.0, unsigned int threshGensSinceImproved = 15, const std::string& stats_filename = "");

		/**
		 * @brief Constructor for the Population class from a file.
		 * @param filename The file path.
		 * @param bias_values The initial biases values (e.g., k-th bias will have value bias_values[k]).
		 * @param resetValues The biases reset values (e.g., k-th bias can be resetted to resetValues[k]).
		 * @param activationFns The activation functions (e.g., activationFns[i][j] is a pointer to an activation function that takes an input of type of index i and return a type of index j output).
		 * @param inputsActivationFns The activation functions of the bias & inputs nodes. The first functions are dedicated to the bias nodes and the other ones to the inputs ones.
		 * @param outputsActivationFns The activation functions of the outputs nodes.
		 * @param logger A pointer to the logger for logging.
		 * @param stats_filename The filename for statistics. (default is an empty string, which doesn't create any file)
		 */
		Population (const std::string& filename, const std::vector<void*>& bias_values, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, const std::vector<ActivationFnBase*> inputsActivationFns, const std::vector<ActivationFnBase*> outputsActivationFns, spdlog::logger* logger, const std::string& stats_filename = "");

		/**
		 * @brief Destructor for the Population class.
		 */
		~Population ();

		//Population (const std::string filename) {load(filename);};

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
		Genome<Types...>& getGenome (int id = -1);

		/**
		 * @brief Get a pointer to the Genome with the specified ID.
		 * @param id The ID of the Genome to retrieve. If set to any negative number, the fitter genome will be returned. (default is -1)
		 * @return A pointer to the Genome with the specified ID.
		 */
		Genome<Types...>* getpGenome (int id = -1);

		/**
		 * @brief Load the inputs for the entire population.
		 * @tparam T_in The type of input data.
		 * @param inputs A vector containing inputs to be loaded.
		 */
		template <typename T_in>
		void loadInputs (std::vector<T_in>& inputs);

		/**
		 * @brief Load a single input for the entire population.
		 * @tparam T_in The type of input data.
		 * @param input The input data to be loaded.
		 * @param input_id The ID of the input to load.
		 */
		template <typename T_in>
		void loadInput (T_in& input, unsigned int input_id);

		/**
		 * @brief Load the inputs for a specific genome.
		 * @tparam T_in The type of input data.
		 * @param inputs A vector containing inputs to be loaded.
		 * @param genome_id The ID of the genome for which to load the inputs.
		 */
		template <typename T_in>
		void loadInputs (std::vector<T_in>& inputs, unsigned int genome_id);

		/**
		 * @brief Load a single input data for a specific genome.
		 * @tparam T_in The type of input data.
		 * @param input The input data to be loaded.
		 * @param input_id The ID of the input to load.
		 * @param genome_id The ID of the genome for which to load the input.
		 */
		template <typename T_in>
		void loadInput (T_in& input, unsigned int input_id, unsigned int genome_id);

		/**
		 * @brief Load the inputs for the entire population.
		 * @param inputs A vector containing inputs to be loaded.
		 */
		void loadInputs (std::vector<void*>& inputs);

		/**
		 * @brief Load a single input for the entire population.
		 * @param input The input data to be loaded.
		 * @param input_id The ID of the input to load.
		 */
		void loadInput (void* input, unsigned int input_id);

		/**
		 * @brief Load the inputs for a specific genome.
		 * @param inputs A vector containing inputs to be loaded.
		 * @param genome_id The ID of the genome for which to load the inputs.
		 */
		void loadInputs (std::vector<void*>& inputs, unsigned int genome_id);

		/**
		 * @brief Load a single input data for a specific genome.
		 * @param input The input data to be loaded.
		 * @param input_id The ID of the input to load.
		 * @param genome_id The ID of the genome for which to load the input.
		 */
		void loadInput (void* input, unsigned int input_id, unsigned int genome_id);

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
		 * @brief Run multiple times the networks over the inputs without taking care of the outputs. The inputs are shared among the genomes. 
		 * @param inputs The inputs.
		 * @param outputs Pointer to the outputs. (default is nullptr which doesn't track any output)
		 * @param maxThreads Maximum number of threads. (default is 0 which default to the number of cores)
		 */
		void run (const std::vector<std::vector<void*>>& inputs, std::vector<std::vector<void*>>* outputs = nullptr, unsigned int maxThreads = 0);

		/**
		 * @brief Run multiple times the networks over the inputs. The inputs are different for each genomes.
		 * @param inputs The inputs.
		 * @param outputs Pointer to the outputs. (default is nullptr which doesn't track any output)
		 * @param maxThreads Maximum number of threads. (default is 0 which default to the number of cores)
		 */
		void run (const std::vector<std::vector<std::vector<void*>>>& inputs, std::vector<std::vector<void*>>* outputs = nullptr, unsigned int maxThreads = 0);

		/**
		 * @brief Run multiple times the networks by looping the outputs and inputs e.g. the n-th outputs is the n+1-th inputs.
		 * @param N_runs The number of networks's runs e.g. the number of loop.
		 * @param outputs Pointer to the outputs. (default is nullptr which doesn't track any output)
		 * @param maxThreads Maximum number of threads. (default is 0 which default to the number of cores)
		 */
		void run (const unsigned int N_runs, std::vector<std::vector<void*>>* outputs = nullptr, unsigned int maxThreads = 0);

		/**
		 * @brief Run the network of the entire population.
		 * @param maxThreads Maximum number of threads. (default is 0 which default to the number of cores)
		 *
		 * Run the network of every genomes of the population. This means computing each node's input and output of each population's genome.
		 * This function use multithreading and should be preffered relatively to `runNetwork (unsigned int genome_id)`.
		 */
		void runNetworks (unsigned int maxThreads = 0);

		/**
		 * @brief Run the network of a specific genome.
		 * @param genome_id The ID of the genome for which to run the newtork.
		 * @return 'false' if the network raised a NaN, 'true' else.
		 *
		 * Run the network of a specific genome. This means computing each node's input and output.
		 * This function does not use multithreading and should be avoid. The use of `runNetwork ()` should be preferred
		 */
		bool runNetwork (unsigned int genome_id);

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
		 * @return The specified output of the specified genome.
		 */
		template <typename T_out>
		T_out getOutput (unsigned int output_id, unsigned int genome_id);

		/**
		 * @brief Get the outputs of a specific genome.
		 * @param genome_id The ID of the genome for which to get the outputs.
		 * @return A vector containing void pointers to the outputs of the specified genome.
		 */
		std::vector<void*> getOutputs (unsigned int genome_id);

		/**
		 * @brief Get a specific output of a specific genome.
		 * @param output_id The ID of the output data.
		 * @param genome_id The ID of the genome for which to get the outputs.
		 * @return A void pointer to the specified output of the specified genome.
		 */
		void* getOutput (unsigned int output_id, unsigned int genome_id);

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
		 * @param speciesSizeEvolutionMax The maximum factor of the species's size evolution. (default is 3.0)
		 * @param speciesSizeEvolutionMin The minimum factor of the species's size evolution. (default is 0.33)
		 * @param speciesSizeLimit The maximum factor, relatively to the target size, of the species's size. (default is 1.75)
		 */
		void speciate (unsigned int target = 5, unsigned int maxIterationsReachTarget = 100, double stepThresh = 0.3, double a = 1.0, double b = 1.0, double c = 0.4, double speciesSizeEvolutionMax = 3.0, double speciesSizeEvolutionMin = 0.33, double speciesSizeLimit = 1.75);

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
		void mutate (const mutationParams_t& params);

		/**
		 * @brief Perform mutation operations for the entire population.
		 * @param paramsMap A function that returns mutation parameters relative to the genome's fitness.
		 */
		void mutate (const std::function<mutationParams_t (double)>& paramsMap);

		/**
		 * @brief Build the next generation. Actually, each new genome is the result of a crossover between two parents from the current generation or a mutation of a genome of the curretnt generation.
		 * @param mutationParams Mutation parameters.
		 * @param elitism Set to true to keep the fitter genome in the new generation. (default is false)
		 * @param crossover_rate The probability of performing crossover for each new genome. (default is 0.9)
		 */
		void buildNextGen (const mutationParams_t& mutationParams, bool elitism = false, double crossover_rate = 0.9);

		/**
		 * @brief Build the next generation. Actually, each new genome is the result of a crossover between two parents from the current generation or a mutation of a genome of the curretnt generation.
		 * @param mutationParamsMap A function that returns mutation parameters relative to the genome's fitness.
		 * @param elitism Set to true to keep the fitter genome in the new generation. (default is false)
		 * @param crossover_rate The probability of performing crossover for each new genome. (default is 0.9)
		 */
		void buildNextGen (const std::function<mutationParams_t (double)>& mutationParamsMap, bool elitism = false, double crossover_rate = 0.9);

		/**
		 * @brief Print information on the population.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (const std::string& prefix = "");

		/**
		 * @brief Draw a graphical representation of the specified genome's network.
		 * @param genome_id The ID of the genome to draw.
		 * @param font_path The filename of the font to be used for labels.
		 * @param windowWidth The width of the drawing window. (default is 1300)
		 * @param windowHeight The height of the drawing window. (default is 800)
		 * @param dotsRadius The radius of the dots representing nodes. (default is 6.5f)
		 */
		void drawGenome (unsigned int genome_id, const std::string& font_path, unsigned int windowWidth = 1300, unsigned int windowHeight = 800, float dotsRadius = 6.5f);

		/**
		 * @brief Save the Population instance to a file.
		 * @param filename The file path.
		 */
		void save (const std::string& filename);

		/**
		 * @brief Load a Population instance from a file.
		 * @param filename ThePopulation file path.
		 */
		void load (const std::string& filename);

		/**
		 * @brief Serialize the Population instance to an output file stream.
		 * @param outFile The output file stream to which the Population instance will be written.
		 */
		void serialize (std::ofstream& outFile);

		/**
		 * @brief Deserialize a Population instance from an input file stream.
		 * @param inFile The input file stream from which the Population instance will be read.
		 */
		void deserialize (std::ifstream& inFile);

		/**
		 * @brief Iterator pointing to the beginning of the genomes.
		 * @return The begin of genomes.
		 */
		auto begin () {return genomes.begin ();}

		/**
		 * @brief Iterator pointing to the end of the genomes.
		 * @return The end of genomes.
		 */
		auto end () {return genomes.end ();}

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
		std::unordered_map <unsigned int, std::unique_ptr<Genome<Types...>>> genomes;
		std::vector<Species<Types...>> species;
		std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
		std::vector<ActivationFnBase*> inputsActivationFns;
		std::vector<ActivationFnBase*> outputsActivationFns;
		innovationConn_t conn_innov;
		innovationNode_t node_innov;	// node's innovation id is more like a global id to decerne two different nodes than something to track innovation

		spdlog::logger* logger;
		std::ofstream statsFile;

		std::unordered_map <unsigned int, Connection> GetWeightedCentroid (unsigned int speciesId);
		void UpdateFitnesses (double speciesSizeEvolutionMax, double speciesSizeEvolutionMin, double speciesSizeLimit, unsigned int NspeciesTarget);
		int SelectParent (unsigned int iSpe);

};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename... Types>
Population<Types...>::Population(unsigned int popSize, const std::vector<size_t>& bias_sch, const std::vector<size_t>& inputs_sch, const std::vector<size_t>& outputs_sch, const std::vector<std::vector<size_t>>& hiddens_sch_init, const std::vector<void*>& bias_values, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, const std::vector<ActivationFnBase*> inputsActivationFns, const std::vector<ActivationFnBase*> outputsActivationFns, unsigned int N_ConnInit, double probRecuInit, double weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger, distanceFn dstType, double speciationThreshInit, unsigned int threshGensSinceImproved, const std::string& stats_filename) :
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
	inputsActivationFns (inputsActivationFns),
	outputsActivationFns (outputsActivationFns),
	logger (logger)
{
	logger->info ("Population initialization");

	if (stats_filename != "") statsFile.open (stats_filename);
	if (statsFile.is_open ()) {
		statsFile << "Generation,Best Fitness,Average Fitness,Average Fitness (Adjusted),Species0,Species1\n";
		statsFile.flush ();
	}

	generation = 0;
	fittergenome_id = -1;
	avgFitness = 0.0;
	avgFitnessAdjusted = 0.0;

	genomes.reserve (popSize);
	for (unsigned int i = 0; i < popSize; i++) {
		genomes.insert (std::make_pair (i, std::make_unique<Genome<Types...>> (i, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_values, resetValues, activationFns, inputsActivationFns, outputsActivationFns, &conn_innov, &node_innov, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger)));
	}
}

template <typename... Types>
Population<Types...>::Population (const std::string& filename, const std::vector<void*>& bias_values, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, const std::vector<ActivationFnBase*> inputsActivationFns, const std::vector<ActivationFnBase*> outputsActivationFns, spdlog::logger* logger, const std::string& stats_filename) :
	bias_values (bias_values),
	resetValues (resetValues),
	activationFns (activationFns),
	inputsActivationFns (inputsActivationFns),
	outputsActivationFns (outputsActivationFns),
	logger (logger)
{
	logger->info ("Population loading");

	if (stats_filename != "") statsFile.open (stats_filename);
	if (statsFile.is_open ()) {
		statsFile << "Generation,Best Fitness,Average Fitness,Average Fitness (Adjusted),Species0,Species1\n";
		statsFile.flush ();
	}

	load (filename);
}

template <typename... Types>
Population<Types...>::~Population () {
	logger->info ("Population destruction");
	if (statsFile.is_open ()) statsFile.close ();
}

template <typename... Types>
Genome<Types...>& Population<Types...>::getGenome (int id) {
	if (id < 0 || id >= (int) popSize) {
		if (fittergenome_id < 0) {
			// fitter genome cannot be found
			logger->warn ("Calling Population<Types...>::getGenome cannot determine which is the more fit genome: in order to know it, call Population<Types...>::speciate first. Returning the first genome.");
			return *genomes [0];
		}
		return *genomes [fittergenome_id];
	}
	return *genomes [id];
}

template <typename... Types>
Genome<Types...>* Population<Types...>::getpGenome (int id) {
	if (id < 0 || id >= (int) popSize) {
		if (fittergenome_id < 0) {
			// fitter genome cannot be found
			logger->warn ("Calling Population<Types...>::getGenome cannot determine which is the more fit genome: in order to know it, call Population<Types...>::speciate first. Returning the first genome.");
			return genomes [0].get ();
		}
		return genomes [fittergenome_id].get ();
	}
	return genomes [id].get ();
}

template <typename... Types>
template <typename T_in>
void Population<Types...>::loadInputs (std::vector<T_in>& inputs) {
	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->template loadInputs<T_in> (inputs);
	}
}

template <typename... Types>
template <typename T_in>
void Population<Types...>::loadInputs (std::vector<T_in>& inputs, unsigned int genome_id) {
	genomes [genome_id]->template loadInputs<T_in> (inputs);
}

template <typename... Types>
template <typename T_in>
void Population<Types...>::loadInput (T_in& input, unsigned int input_id) {
	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->template loadInput<T_in> (input, input_id);
	}
}

template <typename... Types>
template <typename T_in>
void Population<Types...>::loadInput (T_in& input, unsigned int input_id, unsigned int genome_id) {
	genomes [genome_id]->template loadInput<T_in> (input, input_id);
}


template <typename... Types>
void Population<Types...>::loadInputs (std::vector<void*>& inputs) {
	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->loadInputs (inputs);
	}
}

template <typename... Types>
void Population<Types...>::loadInputs (std::vector<void*>& inputs, unsigned int genome_id) {
	genomes [genome_id]->loadInputs (inputs);
}

template <typename... Types>
void Population<Types...>::loadInput (void* input, unsigned int input_id) {
	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->loadInput (input, input_id);
	}
}

template <typename... Types>
void Population<Types...>::loadInput (void* input, unsigned int input_id, unsigned int genome_id) {
	genomes [genome_id]->loadInput (input, input_id);
}

template <typename... Types>
void Population<Types...>::run (const std::vector<std::vector<void*>>& inputs, std::vector<std::vector<void*>>* outputs, unsigned int maxThreads) {
	if (outputs != nullptr) {
		// we do care of outputs

		ThreadPool<std::vector<void*>> pool (maxThreads);

		// the task
		std::function<std::vector<void*> (Genome<Types...>*, std::vector<std::vector<void*>>&)> func = [&] (Genome<Types...>* genome, std::vector<std::vector<void*>>& inputs_gen) -> std::vector<void*> {
			for (std::vector<void*>& inputs_cur : inputs_gen) {
				genome->loadInputs (inputs_cur);
				if (!genome->runNetwork ()) return {};
				genome->saveOutputs ();
			}
			return genome->getSavedOutputs ();
		};

		// fill the thread pool
		std::vector<std::future<std::vector<void*>>> results;
		results.reserve (popSize);
		for (unsigned int i = 0; i < popSize; i++) {
			// add the task to the pool
			results.emplace_back (pool.enqueue (
				func,
				genomes [i].get (),
				inputs
			));
		}

		// wait for tasks end
		for (std::future<std::vector<void*>>& result : results) {
			result.wait ();
		}

		// get results
		outputs->clear ();
		for (std::future<std::vector<void*>>& result : results) {
			outputs->push_back (result.get ());
		}

	} else {
		// we don't care of outputs
		UNUSED (outputs);

		ThreadPool<void> pool (maxThreads);

		// the task
		std::function<void (Genome<Types...>*, std::vector<std::vector<void*>>&)> func = [&] (Genome<Types...>* genome, std::vector<std::vector<void*>>& inputs_gen) -> void {
			for (std::vector<void*>& inputs_cur : inputs_gen) {
				genome->loadInputs (inputs_cur);
				if (!genome->runNetwork ()) return;
			}
		};

		// fill the thread pool
		for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
			// add the task to the pool
			pool.enqueue (
				func,
				genome.second.get (),
				inputs
			);
		}

		// will wait for the end of all tasks as ~ThreadPool () will be called

	}
}

template <typename... Types>
void Population<Types...>::run (const std::vector<std::vector<std::vector<void*>>>& inputs, std::vector<std::vector<void*>>* outputs, unsigned int maxThreads) {
	if (outputs != nullptr) {
		// we do care of outputs

		ThreadPool<std::vector<void*>> pool (maxThreads);

		// the task
		std::function<std::vector<void*> (Genome<Types...>*, std::vector<std::vector<void*>>&)> func = [&] (Genome<Types...>* genome, std::vector<std::vector<void*>>& inputs_gen) -> std::vector<void*> {
			for (std::vector<void*>& inputs_cur : inputs_gen) {
				genome->loadInputs (inputs_cur);
				if (!genome->runNetwork ()) return {};
				genome->saveOutputs ();
			}
			return genome->getSavedOutputs ();
		};

		// fill the thread pool
		std::vector<std::future<std::vector<void*>>> results;
		results.reserve (popSize);
		for (unsigned int i = 0; i < popSize; i++) {
			// add the task to the pool
			results.emplace_back (pool.enqueue (
				func,
				genomes [i].get (),
				inputs [i]
			));
		}

		// wait for tasks end
		for (std::future<std::vector<void*>>& result : results) {
			result.wait ();
		}

		// get results
		outputs->clear ();
		for (std::future<std::vector<void*>>& result : results) {
			outputs->push_back (result.get ());
		}

	} else {
		// we don't care of outputs
		UNUSED (outputs);

		ThreadPool<void> pool (maxThreads);

		// the task
		std::function<void (Genome<Types...>*, std::vector<std::vector<void*>>&)> func = [&] (Genome<Types...>* genome, std::vector<std::vector<void*>>& inputs_gen) -> void {
			for (std::vector<void*>& inputs_cur : inputs_gen) {
				genome->loadInputs (inputs_cur);
				if (!genome->runNetwork ()) return;
			}
		};

		// fill the thread pool
		for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
			// add the task to the pool
			pool.enqueue (
				func,
				genome.second.get (),
				inputs [genome.first]
			);
		}

		// will wait for the end of all tasks as ~ThreadPool () will be called

	}
}

template <typename... Types>
void Population<Types...>::run (const unsigned int N_runs, std::vector<std::vector<void*>>* outputs, unsigned int maxThreads) {
	if (outputs != nullptr) {
		// we do care of outputs

		ThreadPool<std::vector<void*>> pool (maxThreads);

		// the task
		std::function<std::vector<void*> (Genome<Types...>*)> func = [&] (Genome<Types...>* genome) -> std::vector<void*> {
			for (unsigned int k = 0; k < N_runs; k++) {
				genome->loadInputs (genome->getOutputs ());
				if (!genome->runNetwork ()) return {};
				genome->saveOutputs ();
			}
			return genome->getSavedOutputs ();
		};

		// fill the thread pool
		std::vector<std::future<std::vector<void*>>> results;
		results.reserve (popSize);
		for (unsigned int i = 0; i < popSize; i++) {
			// add the task to the pool
			results.emplace_back (pool.enqueue (
				func,
				genomes [i].get ()
			));
		}

		// wait for tasks end
		for (std::future<std::vector<void*>>& result : results) {
			result.wait ();
		}

		// get results
		outputs->clear ();
		for (std::future<std::vector<void*>>& result : results) {
			outputs->push_back (result.get ());
		}

	} else {
		// we don't care of outputs
		UNUSED (outputs);

		ThreadPool<void> pool (maxThreads);

		// the task
		std::function<void (Genome<Types...>*)> func = [&] (Genome<Types...>* genome) -> void {
			for (unsigned int k = 0; k < N_runs; k++) {
				genome->loadInputs (genome->getOutputs ());
				if (genome->runNetwork ()) return;
			}
		};

		// fill the thread pool
		for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
			// add the task to the pool
			pool.enqueue (
				func,
				genome.second.get ()
			);
		}

		// will wait for the end of all tasks as ~ThreadPool () will be called

	}

}

template <typename... Types>
void Population<Types...>::resetMemory () {
	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->resetMemory ();
	}
}

template <typename... Types>
void Population<Types...>::resetMemory (unsigned int genome_id) {
	genomes [genome_id]->resetMemory ();
}

template <typename... Types>
void Population<Types...>::runNetworks (unsigned int maxThreads) {
	ThreadPool<void> pool (maxThreads);

	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		// add the task to the pool
		pool.enqueue (
			[&] () -> void {
				genome.second->runNetwork ();
			}
		);
	}

	// wait for the end of all tasks as ~ThreadPool () is called
}

template <typename... Types>
bool Population<Types...>::runNetwork(unsigned int genome_id) {
	return genomes [genome_id]->runNetwork ();
}

template <typename... Types>
template <typename T_out>
std::vector<T_out> Population<Types...>::getOutputs (unsigned int genome_id) {
	return genomes [genome_id]->template getOutputs<T_out> ();
}

template <typename... Types>
template <typename T_out>
T_out Population<Types...>::getOutput (unsigned int output_id, unsigned int genome_id) {
	return genomes [genome_id]->template getOutput<T_out> (output_id);
}

template <typename... Types>
std::vector<void*> Population<Types...>::getOutputs (unsigned int genome_id) {
	return genomes [genome_id]->getOutputs ();
}

template <typename... Types>
void* Population<Types...>::getOutput (unsigned int output_id, unsigned int genome_id) {
	return genomes [genome_id]->getOutput (output_id);
}

template <typename... Types>
void Population<Types...>::setFitness (double fitness, unsigned int genome_id) {
	genomes [genome_id]->setFitness (fitness);
}

template <typename... Types>
void Population<Types...>::speciate (unsigned int target, unsigned int maxIterationsReachTarget, double stepThresh, double a, double b, double c, double speciesSizeEvolutionMax, double speciesSizeEvolutionMin, double speciesSizeLimit) {
	logger->info ("Speciation");

	std::vector<Species<Types...>> tmpspecies;
	unsigned int nbSpeciesAlive = 0;
	unsigned int ite = 0;

	while (ite < maxIterationsReachTarget && nbSpeciesAlive != target) {

		// init tmpspecies
		tmpspecies.clear ();
		tmpspecies = species;
		nbSpeciesAlive = 0;

		// reset tmpspecies
		for (size_t iSpe = 0; iSpe < tmpspecies.size (); iSpe++) {
			if (!tmpspecies [iSpe].isDead) {	// if the species is still alive
				tmpspecies [iSpe].members.clear ();
			}
		}

		// speciation
		for (unsigned int genome_id = 0; genome_id < popSize; genome_id++) {
			size_t itmpspeciesBest;
			double dstBest = std::numeric_limits<double>::max ();
			if (tmpspecies.size () > 0) {	// if there is at least one species

				// we search for the closest species
				double dst;
				for (size_t itmpspecies = 0; itmpspecies < tmpspecies.size (); itmpspecies++) {
					if (!tmpspecies [itmpspecies].isDead && (dst = tmpspecies [itmpspecies].distanceWith (genomes [genome_id], a, b, c)) <= dstBest) {
						// we found a closer one
						itmpspeciesBest = itmpspecies;
						dstBest = dst;
					}
				}

			}
			if (dstBest >= speciationThresh) {
				// the closest species is too far or there is no species: we create a new one
				itmpspeciesBest = tmpspecies.size ();
				tmpspecies.push_back (Species<Types...> ((unsigned int) itmpspeciesBest, genomes [genome_id]->connections, dstType));
			}

			tmpspecies [itmpspeciesBest].members.push_back (genome_id);
			genomes [genome_id]->speciesId = tmpspecies [itmpspeciesBest].id;
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

			unsigned int leaderId = species [iSpe].members [0];
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
	UpdateFitnesses (speciesSizeEvolutionMax, speciesSizeEvolutionMin, speciesSizeLimit, target);
}

template <typename... Types>
std::unordered_map <unsigned int, Connection> Population<Types...>::GetWeightedCentroid (unsigned int speciesId) {
	std::unordered_map <unsigned int, Connection> result;

	double sumFitness = 0.0;

	for (size_t i = 0; i < species [speciesId].members.size (); i++) {	// for each genome in the species
		double fitness = genomes [species [speciesId].members [i]]->fitness;

		for (const Connection& conn : genomes [species [speciesId].members [i]]->connections) {	// for each of its connections
			if (conn.enabled) {	// only pay attention to active ones
				size_t curResConn = 0;
				size_t result_sz = result.size ();
				while (curResConn < result_sz && result [curResConn].innovId != conn.innovId) {	// while we have not found any corresponding connection in result
					curResConn++;
				}
				if (curResConn >= result_sz) {	// there is no corresponding connections, we add it
					result [curResConn] = conn;
					result [curResConn].weight = 0.0;	// set its weight as null as all the previous genomes doesn't contains it
				}
				result [curResConn].weight += conn.weight * fitness;	// we add the connection's weight dot the genome's fitness (weighted centroid, check below)
			}
		}

		sumFitness += fitness;
	}

	// we divide each weight by sumFitness to have the average (weighted centroid)
	if (sumFitness > 0.0) {
		for (std::pair<unsigned int, Connection>& conn : result) {
			conn.second.weight /= sumFitness;
		}
	} else {
		// null sumFitness
		for (std::pair<unsigned int, Connection>& conn : result) {
			conn.second.weight = std::numeric_limits<double>::max ();
		}
	}

	return result;
}

template <typename... Types>
void Population<Types...>::UpdateFitnesses (double speciesSizeEvolutionMax, double speciesSizeEvolutionMin, double speciesSizeLimit, unsigned int NspeciesTarget) {
	fittergenome_id = 0;
	avgFitness = 0.0;
	avgFitnessAdjusted = 0.0;

	// process avgFitness and found fittergenome_id
	for (const std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		avgFitness += genome.second->fitness;

		if (genome.second->fitness > genomes [fittergenome_id]->fitness) {
			fittergenome_id = genome.first;
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

				double evolutionFactor = species [i].avgFitnessAdjusted / (avgFitnessAdjusted + std::numeric_limits<double>::epsilon ());
				if (evolutionFactor > speciesSizeEvolutionMax) {
					evolutionFactor = speciesSizeEvolutionMax;	// we limit the species evolution factor: a species's size cannot skyrocket from few genomes
				} else {
					if (evolutionFactor < speciesSizeEvolutionMin) {
						evolutionFactor = speciesSizeEvolutionMin;	// we limit the species evolution factor: a species's size cannot be slashed
					}
				}
				int allowedOffspring = (int) ((double) species [i].members.size () * evolutionFactor);	// note that (int) 0.9 == 0.0

				const int sizelimit = (int) ((double) (popSize / NspeciesTarget) * speciesSizeLimit);
				if (allowedOffspring > sizelimit) allowedOffspring = sizelimit;

				species [i].allowedOffspring = allowedOffspring;
			} else {
				// the species cannot have offsprings it has not improved for a long time
				species[i].allowedOffspring = 0;
			}
		}
	}

	// update isDead boolean
	for (size_t i = 0; i < species.size (); i ++) {
		if (species [i].allowedOffspring <= 0) species [i].isDead = true;
	}

	// add satistics to the file
	if (statsFile.is_open ()) {
		statsFile << generation << "," << genomes [fittergenome_id]->fitness << "," << avgFitness << "," << avgFitnessAdjusted << ",";
		for (size_t i = 0; i < species.size () - 1; i ++) {
			statsFile << species [i].members.size () << ",";
		}
		statsFile << species.back ().members.size () << "\n";
		statsFile.flush ();
	}
}

template <typename... Types>
void Population<Types...>::crossover (bool elitism, double crossover_rate) {
	logger->info ("Crossover");
	std::unordered_map<unsigned int, std::unique_ptr<Genome<Types...>>> newGenomes;
	newGenomes.reserve (popSize);

	unsigned int genomeId = 0;

	if (elitism) {	// elitism mode on = we conserve during generations the more fit genome
		newGenomes.insert (std::make_pair (genomeId, genomes [fittergenome_id]->clone ()));
		newGenomes [genomeId]->id = genomeId;
		genomeId++;
	}

	// scale the number of offsprings to the exact population's size
	std::vector<unsigned int> species_alive;
	int N_offsprings = 0; 
	for (unsigned int iSpe = 0; iSpe < (unsigned int) species.size (); iSpe ++) {
		if (!species [iSpe].isDead) {
			N_offsprings += species [iSpe].allowedOffspring;
			species_alive.push_back(iSpe);
		}
	}
	for (int k = 0; k < (int) popSize - N_offsprings - (int) elitism; k++) {
		// some offsprings are missing, let's help the weakest species
		std::sort(species_alive.begin(), species_alive.end(), [&](const unsigned int& a, const unsigned int& b) {return species [a].allowedOffspring < species [b].allowedOffspring;});
		species [species_alive [0]].allowedOffspring += 1;
	}
	for (int k = 0; k < (int) elitism + N_offsprings - (int) popSize; k++) {
		// there is too meny offsprings, let's weaken the strongest species
		std::sort(species_alive.begin(), species_alive.end(), [&](const unsigned int& a, const unsigned int& b) {return species [a].allowedOffspring > species [b].allowedOffspring;});
		species [species_alive [0]].allowedOffspring -= 1;
	}

	// process offsprings
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

					newGenomes.insert (std::make_pair (genomeId, genomes [iMainParent]->clone ()));
					std::unique_ptr<Genome<Types...>>& genome = newGenomes [genomeId];
					genome->id = genomeId;
					genomeId++;

					// connections shared by both of the parents must be randomly wheighted
					for (const std::pair<const unsigned int, Connection>& connMainParent : genomes [iMainParent]->connections) {
						for (const std::pair<const unsigned int, Connection>& connSecondParent : genomes [iSecondParent]->connections) {
							if (connMainParent.second.innovId == connSecondParent.second.innovId) {
								if (Random_Double (0.0, 1.0, true, false) < 0.5) {	// 50 % of chance for each parent, newGenome already have the wheight of MainParent
									genome->connections [connMainParent.second.id].weight = connSecondParent.second.weight;
								}
							}
						}
					}
				} else {
					// the genome is kept for the new generation (there is no crossover which emphasize mutation's effect eg exploration)
					newGenomes.insert (std::make_pair (genomeId, genomes [iParent1]->clone ()));
					newGenomes [genomeId]->id = genomeId;
					genomeId++;
				}
			}
		}
	}

	// replace the current genomes by the new ones
	genomes.clear ();
	genomes = std::move (newGenomes);

	// reset species members
	for (size_t i = 0; i < species.size (); i++) {
		species [i].members.clear ();
		species [i].isDead = true;
	}
	for (const std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		if (genome.second->speciesId > -1) {
			species [genome.second->speciesId].members.push_back (genome.first);
			species [genome.second->speciesId].isDead = false;	// empty species will stay to isDead = true
		}
	}

	fittergenome_id = -1;	// avoid a missuse of fittergenome_id

	generation ++;
}

template <typename... Types>
int Population<Types...>::SelectParent (unsigned int iSpe) {
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

template <typename... Types>
void Population<Types...>::mutate (const mutationParams_t& params) {
	logger->info ("Mutations");
	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->mutate (&conn_innov, &node_innov, params);
	}
}

template <typename... Types>
void Population<Types...>::mutate (const std::function<mutationParams_t (double)>& paramsMap) {
	logger->info ("Mutations");
	for (std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->mutate (&conn_innov, &node_innov, paramsMap (genome.second->fitness));
	}
}


template <typename... Types>
void Population<Types...>::buildNextGen (const mutationParams_t& mutationParams, bool elitism, double crossover_rate)  {
	logger->info ("Build the new generation");
	std::unordered_map<unsigned int, std::unique_ptr<Genome<Types...>>> newGenomes;
	newGenomes.reserve (popSize);

	unsigned int genomeId = 0;

	if (elitism) {	// elitism mode on = we conserve during generations the more fit genome
		newGenomes.insert (std::make_pair (genomeId, genomes [fittergenome_id]->clone ()));
		newGenomes [genomeId]->id = genomeId;
		genomeId++;
	}

	// scale the number of offsprings to the exact population's size
	std::vector<unsigned int> species_alive;
	int N_offsprings = 0; 
	for (unsigned int iSpe = 0; iSpe < (unsigned int) species.size (); iSpe ++) {
		if (!species [iSpe].isDead) {
			N_offsprings += species [iSpe].allowedOffspring;
			species_alive.push_back(iSpe);
		}
	}
	for (int k = 0; k < (int) popSize - N_offsprings - (int) elitism; k++) {
		// some offsprings are missing, let's help the weakest species
		std::sort(species_alive.begin(), species_alive.end(), [&](const unsigned int& a, const unsigned int& b) {return species [a].allowedOffspring < species [b].allowedOffspring;});
		species [species_alive [0]].allowedOffspring += 1;
	}
	for (int k = 0; k < (int) elitism + N_offsprings - (int) popSize; k++) {
		// there is too meny offsprings, let's weaken the strongest species
		std::sort(species_alive.begin(), species_alive.end(), [&](const unsigned int& a, const unsigned int& b) {return species [a].allowedOffspring > species [b].allowedOffspring;});
		species [species_alive [0]].allowedOffspring -= 1;
	}

	// process offsprings
	for (unsigned int iSpe = 0; iSpe < (unsigned int) species.size (); iSpe ++) {
		if (!species [iSpe].isDead) {
			for (int k = 0; k < species [iSpe].allowedOffspring; k++) {

				// choose pseudo-randomly a first parent
				unsigned int iParent1 = SelectParent (iSpe);

				if (Random_Double (0.0, 1.0, true, false) < crossover_rate && species [iSpe].members.size () > 1) {
					// New genome comes from a crossover

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

					newGenomes.insert (std::make_pair (genomeId, genomes [iMainParent]->clone ()));
					std::unique_ptr<Genome<Types...>>& genome = newGenomes [genomeId];
					genome->id = genomeId;
					genomeId++;

					// connections shared by both of the parents must be randomly wheighted
					for (const std::pair<const unsigned int, Connection>& connMainParent : genomes [iMainParent]->connections) {
						for (const std::pair<const unsigned int, Connection>& connSecondParent : genomes [iSecondParent]->connections) {
							if (connMainParent.second.innovId == connSecondParent.second.innovId) {
								if (Random_Double (0.0, 1.0, true, false) < 0.5) {	// 50 % of chance for each parent, newGenome already have the wheight of MainParent
									genome->connections [connMainParent.second.id].weight = connSecondParent.second.weight;
								}
							}
						}
					}
				} else {
					// New genome comes from a mutation

					newGenomes.insert (std::make_pair (genomeId, genomes [iParent1]->clone ()));
					newGenomes [genomeId]->mutate (&conn_innov, &node_innov, mutationParams);
					newGenomes [genomeId]->id = genomeId;
					genomeId++;
				}
			}
		}
	}

	// replace the current genomes by the new ones
	genomes.clear ();
	genomes = std::move (newGenomes);

	// reset species members
	for (size_t i = 0; i < species.size (); i++) {
		species [i].members.clear ();
		species [i].isDead = true;
	}
	for (const std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		if (genome.second->speciesId > -1) {
			species [genome.second->speciesId].members.push_back (genome.first);
			species [genome.second->speciesId].isDead = false;	// empty species will stay to isDead = true
		}
	}

	fittergenome_id = -1;	// avoid a missuse of fittergenome_id

	generation ++;
}


template <typename... Types>
void Population<Types...>::buildNextGen (const std::function<mutationParams_t (double)>& mutationParamsMap, bool elitism, double crossover_rate) {
	logger->info ("Build the new generation");
	std::unordered_map<unsigned int, std::unique_ptr<Genome<Types...>>> newGenomes;
	newGenomes.reserve (popSize);

	unsigned int genomeId = 0;

	if (elitism) {	// elitism mode on = we conserve during generations the more fit genome
		newGenomes.insert (std::make_pair (genomeId, genomes [fittergenome_id]->clone ()));
		newGenomes [genomeId]->id = genomeId;
		genomeId++;
	}

	// scale the number of offsprings to the exact population's size
	std::vector<unsigned int> species_alive;
	int N_offsprings = 0; 
	for (unsigned int iSpe = 0; iSpe < (unsigned int) species.size (); iSpe ++) {
		if (!species [iSpe].isDead) {
			N_offsprings += species [iSpe].allowedOffspring;
			species_alive.push_back(iSpe);
		}
	}
	for (int k = 0; k < (int) popSize - N_offsprings - (int) elitism; k++) {
		// some offsprings are missing, let's help the weakest species
		std::sort(species_alive.begin(), species_alive.end(), [&](const unsigned int& a, const unsigned int& b) {return species [a].allowedOffspring < species [b].allowedOffspring;});
		species [species_alive [0]].allowedOffspring += 1;
	}
	for (int k = 0; k < (int) elitism + N_offsprings - (int) popSize; k++) {
		// there is too meny offsprings, let's weaken the strongest species
		std::sort(species_alive.begin(), species_alive.end(), [&](const unsigned int& a, const unsigned int& b) {return species [a].allowedOffspring > species [b].allowedOffspring;});
		species [species_alive [0]].allowedOffspring -= 1;
	}

	// process offsprings
	for (unsigned int iSpe = 0; iSpe < (unsigned int) species.size (); iSpe ++) {
		if (!species [iSpe].isDead) {
			for (int k = 0; k < species [iSpe].allowedOffspring; k++) {

				// choose pseudo-randomly a first parent
				unsigned int iParent1 = SelectParent (iSpe);

				if (Random_Double (0.0, 1.0, true, false) < crossover_rate && species [iSpe].members.size () > 1) {
					// New genome comes from a crossover

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

					newGenomes.insert (std::make_pair (genomeId, genomes [iMainParent]->clone ()));
					std::unique_ptr<Genome<Types...>>& genome = newGenomes [genomeId];
					genome->id = genomeId;
					genomeId++;

					// connections shared by both of the parents must be randomly wheighted
					for (const std::pair<const unsigned int, Connection>& connMainParent : genomes [iMainParent]->connections) {
						for (const std::pair<const unsigned int, Connection>& connSecondParent : genomes [iSecondParent]->connections) {
							if (connMainParent.second.innovId == connSecondParent.second.innovId) {
								if (Random_Double (0.0, 1.0, true, false) < 0.5) {	// 50 % of chance for each parent, newGenome already have the wheight of MainParent
									genome->connections [connMainParent.second.id].weight = connSecondParent.second.weight;
								}
							}
						}
					}
				} else {
					// New genome comes from a mutation

					newGenomes.insert (std::make_pair (genomeId, genomes [iParent1]->clone ()));
					newGenomes [genomeId]->mutate (&conn_innov, &node_innov, mutationParamsMap (newGenomes [genomeId]->fitness));
					newGenomes [genomeId]->id = genomeId;
					genomeId++;
				}
			}
		}
	}

	// replace the current genomes by the new ones
	genomes.clear ();
	genomes = std::move (newGenomes);

	// reset species members
	for (size_t i = 0; i < species.size (); i++) {
		species [i].members.clear ();
		species [i].isDead = true;
	}
	for (const std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		if (genome.second->speciesId > -1) {
			species [genome.second->speciesId].members.push_back (genome.first);
			species [genome.second->speciesId].isDead = false;	// empty species will stay to isDead = true
		}
	}

	fittergenome_id = -1;	// avoid a missuse of fittergenome_id

	generation ++;
}


template <typename... Types>
void Population<Types...>::print (const std::string& prefix) {
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
	for (const std::pair<const unsigned int, std::unique_ptr<Genome<Types...>>>& genome : genomes) {
		genome.second->print (prefix + "   ");
		std::cout << std::endl;
	}
	std::cout << prefix << "Species: " << std::endl;
	for (const Species<Types...>& spe : species) {
		spe.print (prefix + "   ");
		std::cout << std::endl;
	}
}

template <typename... Types>
void Population<Types...>::drawGenome (unsigned int genome_id, const std::string& font_path, unsigned int windowWidth, unsigned int windowHeight, float dotsRadius) {
	logger->info ("Drawing genome{}'s network", genome_id);
	genomes [genome_id]->draw (font_path, windowWidth, windowHeight, dotsRadius);
}


template <typename... Types>
void Population<Types...>::serialize (std::ofstream& outFile) {
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
	for (unsigned int k = 0; k < (unsigned int) genomes.size (); k++) {
		unsigned int iGenome = 0;
		while (iGenome < (unsigned int) genomes.size () && genomes [iGenome]->id != k) {
			iGenome++;
		}
		if (iGenome < (unsigned int) genomes.size ()) {
			genomes [iGenome]->serialize (outFile);
		} else {
			// impossible state
		}
	}

    Serialize (species.size (), outFile);
	for (const Species<Types...>& spe : species) {
		spe.serialize (outFile);
	}

	conn_innov.serialize (outFile);
	node_innov.serialize (outFile);
}

template <typename... Types>
void Population<Types...>::deserialize (std::ifstream& inFile) {
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

	size_t sz;

	Deserialize (sz, inFile);
	genomes.clear ();
	genomes.reserve (sz);
	for (unsigned int i = 0; i < (unsigned int) sz; i++) {
		genomes.insert (std::make_pair (i, std::make_unique<Genome<Types...>> (inFile, resetValues, activationFns, inputsActivationFns, outputsActivationFns, logger)));
	}

	Deserialize (sz, inFile);
	species.clear ();
	species.reserve (sz);
	for (size_t i = 0; i < sz; i++) {
		species.push_back (Species<Types...> (inFile));
	}

	conn_innov.deserialize (inFile);
	node_innov.deserialize (inFile);
}

template <typename... Types>
void Population<Types...>::save (const std::string& filename) {
	std::ofstream outFile(filename, std::ios::binary);
	if (!outFile) {
		logger->error ("Cannot open file {} for writing.", filename);
		return;
	}

	serialize (outFile);

	outFile.close ();
}

template <typename... Types>
void Population<Types...>::load (const std::string& filename) {
	std::ifstream inFile(filename, std::ios::binary);
	if (!inFile) {
		logger->error ("Cannot open file {} for reading.", filename);
		return;
	}

	deserialize (inFile);

	inFile.close ();
}

#endif	// POPULATION_HPP
