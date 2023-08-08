#ifndef GENOME_HPP
#define GENOME_HPP

#include <PNEATM/Node/node_base.hpp>
#include <PNEATM/Node/innovation_node.hpp>
#include <PNEATM/Connection/connection.hpp>
#include <PNEATM/Connection/innovation_connection.hpp>
#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
#include <PNEATM/Node/create_node.hpp>
#include <PNEATM/utils.hpp>
#include <vector>
#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <cstring>
#include <spdlog/spdlog.h>
#include <memory>
#include <fstream>


/* HEADER */

namespace pneatm {

/**
 * @brief Mutation parameters used to control mutations.
 */
typedef struct mutationParams {
	/**
	 * @brief Mutations parameters to add a node.
	 */
	struct Nodes {
		/**
		 * @brief The chance of having a node added.
		 *
		 * The value of `rate` typically lies between 0.0 and 1.0, where 0.0 indicates that no nodes will be added, and
		 * 1.0 means that a node will be added in every possible instance.
		 */
		double rate;

		/**
		 * @brief The chance of having a monotyped node added.
		 *
		 * Once a node is added, it has a chance of being monotyped instead of bityped. A monotyped node is a node where
		 * input and ouput types are the same.
		 * The value of `rate` typically lies between 0.0 and 1.0, where 0.0 indicates that a bityped will be added, and
		 * 1.0 means that a monotyped node will be added in every possible instance.
		 */
		double monotypedRate;

		/**
		 * @brief Mutations parameters to add a monotyped node.
		 */
		struct Monotyped {
			/**
			 * @brief The maximum number of iteration to find a valid connection.
			 *
			 * The main interest of this variable is just to avoid having an infinite loop.
			 */
			unsigned int maxIterationsFindConnection;
		};
		/**
		 * @brief Mutations parameters to add a monotyped node.
		 */
		struct Monotyped monotyped;

		/**
		 * @brief Mutations parameters to add a bityped node.
		 */
		struct Bityped {
			/**
			 * @brief The maximum recurrency level of the connection that point to the new node.
			 */
			unsigned int maxRecurrencyEntryConnection;

			/**
			 * @brief The maximum number of iteration to find a valid node.
			 *
			 * The main interest of this variable is just to avoid having an infinite loop.
			 */
			unsigned int maxIterationsFindNode;
		};
		/**
		 * @brief Mutations parameters to add a bityped node.
		 */
		struct Bityped bityped;
	};
	/**
	 * @brief Mutations parameters to add a node.
	 */
	struct Nodes nodes;

	/**
	 * @brief Mutations parameters to change activation functions's parameters.
	 */
	struct Activation_Functions {
		/**
		 * @brief The mutation ratio of activation functions.
		 *
		 * The value of `rate` typically lies between 0.0 and 1.0, where 0.0 indicates that no activation functions will be mutated, and
		 * 1.0 means that every activation functions will be mutated in every possible instance.
		 */
		double rate;
	};
	/**
	 * @brief Mutations parameters to change activation functions's parameters.
	 */
	struct Activation_Functions activation_functions;

	/**
	 * @brief Mutations parameters to add a connection.
	 */
	struct Connections {
		/**
		 * @brief The chance of having a connection added.
		 *
		 * The value of `rate` typically lies between 0.0 and 1.0, where 0.0 indicates that no activation functions will be mutated, and
		 * 1.0 means that a every activation functions will be mutated in every possible instance.
		 */
		double rate;

		/**
		 * @brief The chance of reactivate a disabled connection.
		 *
		 * If after trying to add a connection we found a disabled one, there is a chance of reactivate it.
		 * The value of `reactivateRate` typically lies between 0.0 and 1.0, where 0.0 indicates that the connection will not be reactivaded, and
		 * 1.0 means that it will be reactivaded in every possible instance.
		 */
		double reactivateRate;

		/**
		 * @brief The maximum recurrency level of the connection.
		 */
		unsigned int maxRecurrency;

		/**
		 * @brief The maximum number of iteration to find a valid connection.
		 *
		 * The main interest of this variable is just to avoid having an infinite loop.
		 */
		unsigned int maxIterations;

		/**
		 * @brief The maximum number of iteration to find a valid node.
		 *
		 * The main interest of this variable is just to avoid having an infinite loop.
		 */
		unsigned int maxIterationsFindNode;
	};
	/**
	 * @brief Mutations parameters to add a connection.
	 */
	struct Connections connections;

	/**
	 * @brief Mutations parameters to change the weights.
	 */
	struct Weights {
		/**
		 * @brief The mutation ratio of weights.
		 *
		 * The value of `rate` typically lies between 0.0 and 1.0, where 0.0 indicates that no weights will be mutated, and
		 * 1.0 means that every weights will be mutated in every possible instance.
		 */
		double rate;

		/**
		 * @brief The chance of fully change the weight instead of perturbing it.
		 *
		 * Once a weight is mutated, there is a chance of being re-initialized instead of perturbed.
		 * The value of `fullChangeRate` typically lies between 0.0 and 1.0, where 0.0 indicates that weights will simply be perturbaded, and
		 * 1.0 means that every weights will be re-initialized in every possible instance.
		 */
		double fullChangeRate;

		/**
		 * @brief The perturbation factor.
		 *
		 * e.g. `perturbationFactor = 0.2` means that the `new_weight = current_weight +- x%` with x in [0.0, 20.0]
		 * aka the weight is at most changed of 20%
		 */
		double perturbationFactor;
	};
	/**
	 * @brief Mutations parameters to change the weights.
	 */
	struct Weights weights;
} mutationParams_t;

/**
 * @brief A template class representing a genome.
 * @tparam Args Variadic template arguments that contains all the manipulated types.
 */
template <typename... Args>
class Genome {
	public:
		/**
		 * @brief Constructor for the Genome class.
		 * @param bias_sch The biases scheme (e.g., there is bias_sch[k] biases for type of index k).
		 * @param inputs_sch The inputs scheme (e.g., there is inputs_sch[k] inputs for type of index k).
		 * @param outputs_sch The outputs scheme (e.g., there is outputs_sch[k] outputs for type of index k).
		 * @param hiddens_sch_init The initial hidden nodes scheme (e.g., there is hiddens_sch_init[i][j] hidden nodes of input type of index i and output type of index j).
		 * @param bias_values The initial biases values (e.g., k-th bias will have value bias_values[k]).
		 * @param resetValues The biases reset values (e.g., k-th bias can be resetted to resetValues[k]).
		 * @param activationFns The activation functions (e.g., activationFns[i][j] is a pointer to an activation function that takes an input of type of index i and return a type of index j output).
		 * @param conn_innov A pointer to the connections innovation tracker.
		 * @param node_innov A pointer to the nodes innovation tracker.
		 * @param N_ConnInit The initial number of connections.
		 * @param probRecuInit The initial probability of recurrence.
		 * @param weightExtremumInit The initial weight extremum.
		 * @param maxRecuInit The maximum recurrence value.
		 * @param logger A pointer to the logger for logging.
		 */
		Genome (const std::vector<size_t>& bias_sch, const std::vector<size_t>& inputs_sch, const std::vector<size_t>& outputs_sch, const std::vector<std::vector<size_t>>& hiddens_sch_init, const std::vector<void*>& bias_values, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int N_ConnInit, double probRecuInit, double weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger);

		/**
		 * @brief Constructor for the Genome class. This constructor will not initialized any network.
		 * @param nbBias The number of bias node.
		 * @param nbInput The number of input node.
		 * @param nbOutput The number of output node.
		 * @param N_types The number of types involved in the network (the number of types in the variadic template Args).
		 * @param resetValues The biases reset values (e.g., k-th bias can be resetted to resetValues[k]).
		 * @param activationFns The activation functions (e.g., activationFns[i][j] is a pointer to an activation function that takes an input of type of index i and return a type of index j output).
		 * @param weightExtremumInit The initial weight extremum.
		 * @param logger A pointer to the logger for logging.
		 */
		Genome (unsigned int nbBias, unsigned int nbInput, unsigned int nbOutput, unsigned int N_types, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, double weightExtremumInit, spdlog::logger* logger);

		/**
		 * @brief Constructor for the Genome class from an input file stream.
		 * @param inFile The input file stream.
		 * @param resetValues The biases reset values (e.g., k-th bias can be resetted to resetValues[k]).
		 * @param activationFns The activation functions (e.g., activationFns[i][j] is a pointer to an activation function that takes an input of type of index i and return a type of index j output).
		 * @param logger A pointer to the logger for logging.
		 */
		Genome (std::ifstream& inFile, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, spdlog::logger* logger);

		/**
		 * @brief Destructor for the Genome class.
		 */
		~Genome ();

		/**
		 * @brief Get the fitness.
		 * @return The fitness.
		 */
		double getFitness () {return fitness;};

		/**
		 * @brief Get the species ID which belong the genome.
		 * @return The species ID which belong the genome.
		 */
		int getSpeciesId () {return speciesId;};

		/**
		 * @brief Load the inputs.
		 * @tparam T_in The type of input data.
		 * @param inputs A vector containing inputs to be loaded.
		 */
		template <typename T_in>
		void loadInputs (const std::vector<T_in>& inputs);

		/**
		 * @brief Load an input.
		 * @tparam T_in The type of input data.
		 * @param input The input to be loaded.
		 * @param input_id The ID of the input to load.
		 */
		template <typename T_in>
		void loadInput (T_in& input, int input_id);

		/**
		 * @brief Reset the memory.
		 */
		void resetMemory ();

		/**
		 * @brief Run the network.
		 */
		void runNetwork ();

		/**
		 * @brief Get the outputs.
		 * @tparam T_out The type of output data.
		 * @return A vector containing the outputs.
		 */
		template <typename T_out>
		std::vector<T_out> getOutputs ();

		/**
		 * @brief Get a specific output.
		 * @tparam T_out The type of output data.
		 * @param output_id The ID of the output to get.
		 * @return The ouput.
		 */
		template <typename T_out>
		T_out getOutput (int output_id);

		/**
		 * @brief Perform mutation operations.
		 * @param conn_innov A pointer to the connections innovation tracker.
		 * @param node_innov A pointer to the nodes innovation tracker.
		 * @param params Mutation parameters.
		 */
		void mutate (innovationConn_t* conn_innov, innovationNode_t* node_innov, const mutationParams_t& params);

		/**
		 * @brief Get a clone of the genome.
		 * @return A unique pointer to the created clone of the genome.
		 */
		std::unique_ptr<Genome<Args...>> clone ();

		/**
		 * @brief Print information on the genome.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (const std::string& prefix = "");

		/**
		 * @brief Draw a graphical representation of the network.
		 * @param font_path The filepath of the font to be used for labels.
		 * @param windowWidth The width of the drawing window. (default is 1300)
		 * @param windowHeight The height of the drawing window. (default is 800)
		 * @param dotsRadius The radius of the dots representing nodes. (default is 6.5f)
		 */
		void draw (const std::string& font_path, unsigned int windowWidth = 1300, unsigned int windowHeight = 800, float dotsRadius = 6.5f);

		/**
		 * @brief Serialize the Genome instance to an output file stream.
		 * @param outFile The output file stream to which the Genome instance will be written.
		 */
		void serialize (std::ofstream& outFile);

		/**
		 * @brief Deserialize a Genome instance from an input file stream.
		 * @param inFile The input file stream from which the Genome instance will be read.
		 */
		void deserialize (std::ifstream& inFile);

	private:
		unsigned int nbBias;
		unsigned int nbInput;
		unsigned int nbOutput;
		double weightExtremumInit;
		unsigned int N_types;
		std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
		std::vector<void*> resetValues;

		std::vector <std::unique_ptr<NodeBase>> nodes;
		std::vector <Connection> connections;

		double fitness;
		int speciesId;
		unsigned int N_runNetwork;

		spdlog::logger* logger;

		unsigned int RepetitionNodeCheck (unsigned int index_T_in, unsigned int index_T_out, unsigned int index_activation_fn);
		bool CheckNewConnectionValidity (unsigned int inNodeId, unsigned int outNodeId, unsigned int inNodeRecu, int* disabled_conn_id = nullptr);
		bool CheckNewConnectionCircle (unsigned int inNodeId, unsigned int outNodeId);
		void MutateWeights (double mutateWeightThresh, double mutateWeightFullChangeThresh, double mutateWeightFactor);
		void MutateActivationFn (double rate);
		bool AddConnection (innovationConn_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindConnectionThresh, double reactivateConnectionThresh);
		bool AddMonotypedNode (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxIterationsFindConnectionThresh);
		bool AddBitypedNode (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindNodeThresh);
		void UpdateLayers (int nodeId);
		void UpdateLayers_Recursive (unsigned int nodeId);

	template <typename... Args2>
	friend class Population;
	template <typename... Args2>
	friend class Species;
};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename... Args>
Genome<Args...>::Genome (const std::vector<size_t>& bias_sch, const std::vector<size_t>& inputs_sch, const std::vector<size_t>& outputs_sch, const std::vector<std::vector<size_t>>& hiddens_sch_init, const std::vector<void*>& bias_values, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int N_ConnInit, double probRecuInit, double weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger) :
	weightExtremumInit (weightExtremumInit),
	activationFns (activationFns),
	resetValues (resetValues),
	logger (logger)
{
	logger->trace ("Genome initialization");

	N_types = (unsigned int) activationFns.size ();
	speciesId = -1;
	fitness = 0.0;
	N_runNetwork = 0;

	// NODES
	// bias
	nbBias = 0;
	for (size_t i = 0; i < bias_sch.size (); i++) {
		for (size_t k = 0; k < bias_sch [i]; k++) {
			// get Node<T_in, T_out>
			nodes.push_back(CreateNode::get<Args...> (i, i));

			// setup the node
			nodes.back ()->id = nbBias;
			nodes.back ()->layer = 0;
			nodes.back ()->index_T_in = (unsigned int) i;
			nodes.back ()->index_T_out = (unsigned int) i;
			nodes.back ()->index_activation_fn = 0;	// identity function is set on the first position
			nodes.back ()->setActivationFn (
				activationFns [nodes.back ()->index_T_in][nodes.back ()->index_T_out][nodes.back ()->index_activation_fn]->clone (false)	//activation function with new fresh parameters
			);
			nodes.back ()->innovId = node_innov->getInnovId (
				nodes.back ()->index_T_in,
				nodes.back ()->index_T_out,
				nodes.back ()->index_activation_fn,
				RepetitionNodeCheck (nodes.back ()->index_T_in, nodes.back ()->index_T_out, nodes.back ()->index_activation_fn) - 1
			);
			nodes.back ()->setResetValue (resetValues [i]);	// useless as bias nodes are never resetted
			nodes.back ()->loadInput (bias_values [i]);	// load input now as it will always be the same
			nodes.back ()->process ();	// load output now as it will always be the same

			nbBias ++;
		}
	}
	// input
	nbInput = 0;
	for (size_t i = 0; i < inputs_sch.size (); i++) {
		for (size_t k = 0; k < inputs_sch [i]; k++) {
			// get Node<T_in, T_out>
			nodes.push_back(CreateNode::get<Args...> (i, i));

			// setup the node
			nodes.back ()->id = nbBias + nbInput;
			nodes.back ()->layer = 0;
			nodes.back ()->index_T_in = (unsigned int) i;
			nodes.back ()->index_T_out = (unsigned int) i;
			nodes.back ()->index_activation_fn = 0;	// identity function is set on the first position
			nodes.back ()->setActivationFn (
				activationFns [nodes.back ()->index_T_in][nodes.back ()->index_T_out][nodes.back ()->index_activation_fn]->clone (false)	//activation function with new fresh parameters
			);
			nodes.back ()->innovId = node_innov->getInnovId (
				nodes.back ()->index_T_in,
				nodes.back ()->index_T_out,
				nodes.back ()->index_activation_fn,
				RepetitionNodeCheck (nodes.back ()->index_T_in, nodes.back ()->index_T_out, nodes.back ()->index_activation_fn) - 1
			);
			nodes.back ()->setResetValue (resetValues [i]);

			nbInput ++;
		}
	}
	// output
	nbOutput = 0;
	int outputLayer;
	if (hiddens_sch_init.size () > 0) {
		outputLayer = 2;
	} else {
		outputLayer = 1;
	}
	for (size_t i = 0; i < outputs_sch.size (); i++) {
		for (size_t k = 0; k < outputs_sch [i]; k++) {
			// get Node<T_in, T_out>
			nodes.push_back(CreateNode::get<Args...> (i, i));

			// setup the node
			nodes.back ()->id = nbBias + nbInput + nbOutput;
			nodes.back ()->layer = outputLayer;
			nodes.back ()->index_T_in = (unsigned int) i;
			nodes.back ()->index_T_out = (unsigned int) i;
			nodes.back ()->index_activation_fn = 0;	// identity function is set on the first position
			nodes.back ()->setActivationFn (
				activationFns [nodes.back ()->index_T_in][nodes.back ()->index_T_out][nodes.back ()->index_activation_fn]->clone (false)	//activation function with new fresh parameters
			);
			nodes.back ()->innovId = node_innov->getInnovId (
				nodes.back ()->index_T_in,
				nodes.back ()->index_T_out,
				nodes.back ()->index_activation_fn,
				RepetitionNodeCheck (nodes.back ()->index_T_in, nodes.back ()->index_T_out, nodes.back ()->index_activation_fn) - 1
			);
			nodes.back ()->setResetValue (resetValues [i]);

			nbOutput ++;
		}
	}
	// hidden
	unsigned int nbHidden = 0;
	for (size_t i = 0; i < hiddens_sch_init.size (); i++) {
		for (size_t j = 0; j < hiddens_sch_init [i].size (); j++) {
			for (size_t k = 0; k < hiddens_sch_init [i][j]; k++) {
				// get Node<T_in, T_out>
				nodes.push_back(CreateNode::get<Args...> (i, j));

				// setup the node
				nodes.back ()->id = nbBias + nbInput + nbOutput + nbHidden;
				nodes.back ()->layer = 1;
				nodes.back ()->index_T_in = (unsigned int) i;
				nodes.back ()->index_T_out = (unsigned int) j;
				nodes.back ()->index_activation_fn = Random_UInt (0, (unsigned int) activationFns [i][j].size () - 1);
				nodes.back ()->setActivationFn (
					activationFns [nodes.back ()->index_T_in][nodes.back ()->index_T_out][nodes.back ()->index_activation_fn]->clone (false)	//activation function with new fresh parameters
				);
				nodes.back ()->innovId = node_innov->getInnovId (
					nodes.back ()->index_T_in,
					nodes.back ()->index_T_out,
					nodes.back ()->index_activation_fn,
					RepetitionNodeCheck (nodes.back ()->index_T_in, nodes.back ()->index_T_out, nodes.back ()->index_activation_fn) - 1
				);
				nodes.back ()->setResetValue (resetValues [i]);

				nbHidden ++;
			}
		}
	}

	// CONNECTIONS
	unsigned int iConn = 0;
	while (iConn < N_ConnInit) {

		// inNodeId and outNodeId
		unsigned int inNodeId = Random_UInt (0, (unsigned int) nodes.size() - 1);
		unsigned int outNodeId = Random_UInt (0, (unsigned int) nodes.size() - 1);

		// inNodeRecu
		unsigned int inNodeRecu = 0;
		if (Random_Double (0.0, 1.0, true, false) < probRecuInit) {
			inNodeRecu = Random_UInt (0, maxRecuInit);
		}
		if (CheckNewConnectionValidity (inNodeId, outNodeId, inNodeRecu)) {	// we don't care of former connections as there is no disabled connection for now
			// innovId
			const unsigned int innov_id = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);

			// weight
			const double weight = Random_Double (- weightExtremumInit, weightExtremumInit);

			connections.push_back (Connection (innov_id, inNodeId, outNodeId, inNodeRecu, weight, true));

			// update layers if needed
			if (inNodeRecu == 0 && nodes [outNodeId]->layer <= nodes [inNodeId]->layer) {
				nodes [outNodeId]->layer = nodes [inNodeId]->layer + 1;
				UpdateLayers (outNodeId);
			}

			iConn ++;
		}
	}
}

template <typename... Args>
Genome<Args...>::Genome (unsigned int nbBias, unsigned int nbInput, unsigned int nbOutput, unsigned int N_types, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, double weightExtremumInit, spdlog::logger* logger) :
	nbBias (nbBias),
	nbInput (nbInput),
	nbOutput (nbOutput),
	weightExtremumInit (weightExtremumInit),
	N_types (N_types),
	activationFns (activationFns),
	resetValues (resetValues),
	logger (logger)
{
	logger->trace ("Genome initialization");
	speciesId = -1;
	fitness = 0.0;
	N_runNetwork = 0;
}

template <typename... Args>
Genome<Args...>::Genome (std::ifstream& inFile, const std::vector<void*>& resetValues, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns, spdlog::logger* logger) :
	activationFns (activationFns),
	resetValues (resetValues),
	logger (logger)
{
	logger->trace ("Genome loading");

	deserialize (inFile);
}


template <typename... Args>
Genome<Args...>::~Genome () {
	logger->trace ("Genome destruction");
}

template <typename... Args>
template <typename T_in>
void Genome<Args...>::loadInputs (const std::vector<T_in>& inputs) {
	for (unsigned int i = 0; i < nbInput; i++) {
		nodes [i + nbBias]->loadInput (static_cast<void*> (&inputs [i]));
	}
}

template <typename... Args>
template <typename T_in>
void Genome<Args...>::loadInput (T_in& input, int input_id) {
	nodes [input_id + nbBias]->loadInput (static_cast<void*> (&input));
}

template <typename... Args>
void Genome<Args...>::resetMemory () {
	N_runNetwork = 0;
	for (const std::unique_ptr<NodeBase>& node : nodes) {
		node->reset (true);
	}
}

template <typename... Args>
void Genome<Args...>::runNetwork() {
	/* Process all input and output. For that, it "scans" each layer from the inputs to the last hidden's layer to calculate input with already known value. */ 

	N_runNetwork++;

	// reset input
	for (size_t i = nbBias + nbInput; i < nodes.size(); i++) {
		nodes [i]->reset (false);
	}

	// process nodes[*]->output for input/bias nodes
	for (unsigned int i = 0; i < nbBias + nbInput; i++) {
		nodes [i]->process ();
	}

	int lastLayer = nodes [nbBias + nbInput]->layer;

	for (int ilayer = 1; ilayer <= lastLayer; ilayer++) {
		// process nodes[*]->input
		for (size_t i = 0; i < connections.size (); i++) {
			if (connections [i].enabled && nodes [connections [i].outNodeId]->layer == ilayer) {	// if the connections still exist and is pointing to the current layer
				if (connections [i].inNodeRecu < N_runNetwork) {

					unsigned int depth = connections [i].inNodeRecu;
					// if the connection's input node is further in the network it has not been processed and so its previous output is the current one for it!
					if (nodes [connections [i].inNodeId]->layer >= nodes [connections [i].outNodeId]->layer) depth--;

					nodes [connections [i].outNodeId]->AddToInput (
						nodes [connections [i].inNodeId]->getOutput (depth),
						connections [i].weight
					);
				} else {
					// the input of the connection isn't existing yet!
					// we consider that the connection isn't existing
				}
			}
		}

		// process nodes[*]->output & store outputs
		for (size_t i = 0; i < nodes.size (); i++) {
			if (nodes [i]->layer == ilayer) {
				nodes [i]->process ();
			}
		}
	}
}

template <typename... Args>
template <typename T_out>
std::vector<T_out> Genome<Args...>::getOutputs () {
	std::vector<T_out> outputs;
	for (unsigned int i = 0; i < nbOutput; i++) {
		outputs.push_back (*static_cast<T_out*> (nodes [nbBias + nbInput + i]->getOutput ()));
	}
	return outputs;
}

template <typename... Args>
template <typename T_out>
T_out Genome<Args...>::getOutput (int output_id) {
	return *static_cast<T_out*> (nodes[nbBias + nbInput + output_id]->getOutput ());
}

template <typename... Args>
void Genome<Args...>::mutate(innovationConn_t* conn_innov, innovationNode_t* node_innov, const mutationParams_t& params) {
	// WEIGHTS
	MutateWeights (params.weights.rate, params.weights.fullChangeRate, params.weights.perturbationFactor);

	// ACTIVATION FUNCTIONS
	MutateActivationFn (params.activation_functions.rate);

	// NODES
	if (Random_Double (0.0f, 1.0f, true, false) < params.nodes.rate) {
		if (Random_Double (0.0f, 1.0f, true, false) < params.nodes.monotypedRate) {
			AddMonotypedNode (conn_innov, node_innov, params.nodes.monotyped.maxIterationsFindConnection);
		} else {
			AddBitypedNode (conn_innov, node_innov, params.nodes.bityped.maxRecurrencyEntryConnection, params.nodes.bityped.maxIterationsFindNode);
		}
	}

	// CONNECTIONS
	if (Random_Double (0.0f, 1.0f, true, false) < params.connections.rate) {
		AddConnection (conn_innov, params.connections.maxRecurrency, params.connections.maxIterationsFindNode, params.connections.reactivateRate);
	}
}

template <typename... Args>
unsigned int Genome<Args...>::RepetitionNodeCheck (unsigned int index_T_in, unsigned int index_T_out, unsigned int index_activation_fn) {
	unsigned int c = 0;
	for (const std::unique_ptr<NodeBase>& node : nodes) {
		if (node->index_T_in == index_T_in && node->index_T_out == index_T_out && node->index_activation_fn == index_activation_fn) {
			// the same node is in nodes
			c++;
		}
	}
	return c;
}


template <typename... Args>
bool Genome<Args...>::CheckNewConnectionValidity (unsigned int inNodeId, unsigned int outNodeId, unsigned int inNodeRecu, int* disabled_conn_id) {
	if (nodes [inNodeId]->index_T_out != nodes [outNodeId]->index_T_in) return false;	// connections must link two same objects
	if (outNodeId < nbBias + nbInput) return false;	// connections cannot point to an input node

	for (size_t i = 0; i < connections.size (); i++) {
		if (
			connections[i].inNodeId == inNodeId
			&& connections[i].outNodeId == outNodeId
			&& connections[i].inNodeRecu == inNodeRecu
		) {
			if (connections[i].enabled) {
				return false;	// it is already an enabled connection
			} else {
				// it is a disabled connection
				 *disabled_conn_id = (int) i;
			}
		}
	}
	
	if (inNodeRecu > 0) {
		// if it is a recurrent connection, the unique condition is to not be a copy of another one
		// as a recurrent connection cannot create a circle
		return true;
	}

	// the new connection should not have an output as inNode
	// because if this is the case, outNode's layer > inNode's one (wich is the maximum layer, it is the outputs one)
	if (inNodeId >= nbBias + nbInput && inNodeId < nbBias + nbInput + nbOutput) {
		return false;
	}

	// the new connection should not create a circle in the network
	// because if this is the case, even after updating layers, we can find a connection that cannot be in the regular direction
	if (CheckNewConnectionCircle (inNodeId, outNodeId)) {
		return false;	// the connection will create a connection's circle in the network
	}

	return true;	// test passed well: it is a valid connection!
}

template <typename... Args>
bool Genome<Args...>::CheckNewConnectionCircle (unsigned int inNodeId, unsigned int outNodeId) {
	if (inNodeId == outNodeId) {
		return true;
	}
	for (size_t iConn = 0; iConn < connections.size (); iConn ++) {
		if (connections [iConn].inNodeId == outNodeId && connections [iConn].inNodeRecu == 0) {
			if (CheckNewConnectionCircle (inNodeId, connections [iConn].outNodeId)) {
				return true;
			}
		}
	}
	return false;

}

template <typename... Args>
void Genome<Args...>::MutateWeights (double mutateWeightThresh, double mutateWeightFullChangeThresh, double mutateWeightFactor) {
	logger->trace ("mutation of weights");
	for (size_t i = 0; i < connections.size (); i++) {
		if (connections [i].enabled && Random_Double (0.0f, 1.0f, true, false) < mutateWeightThresh) {
			if (Random_Double (0.0f, 1.0f, true, false) < mutateWeightFullChangeThresh) {
				// reset weight
				connections [i].weight = Random_Double (- weightExtremumInit, weightExtremumInit);
			} else {
				// pertub weight
				connections [i].weight += connections [i].weight * Random_Double (- mutateWeightFactor, mutateWeightFactor);
			}
		}
	}
}

template <typename... Args>
void Genome<Args...>::MutateActivationFn (double rate) {
	logger->trace ("mutation of activation functions");
	for (size_t i = 0; i < nodes.size (); i++) {
		if (Random_Double (0.0f, 1.0f, true, false) < rate) {
			nodes [i]->mutate (fitness);
		}
	}
}

template <typename... Args>
bool Genome<Args...>::AddConnection (innovationConn_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindConnectionThresh, double reactivateConnectionThresh) {
	logger->trace ("adding a new connection");

	// find valid node pair
	unsigned int iterationNb = 1;
	unsigned int inNodeId = Random_UInt (0, (unsigned int) nodes.size() - 1);
	unsigned int outNodeId = Random_UInt (0, (unsigned int) nodes.size() - 1);
	unsigned int inNodeRecu = Random_UInt (0, maxRecurrency);
	int disabled_conn_id = -1;
	while (
		iterationNb < maxIterationsFindConnectionThresh
		&& !CheckNewConnectionValidity (inNodeId, outNodeId, inNodeRecu, &disabled_conn_id)
	) {
		inNodeId = Random_UInt (0, (unsigned int) nodes.size() - 1);
		outNodeId = Random_UInt (0, (unsigned int) nodes.size() - 1);
		inNodeRecu = Random_UInt (0, maxRecurrency);
		iterationNb ++;
	}
	
	if (iterationNb < maxIterationsFindConnectionThresh) {	// a valid connection has been found
		// mutation
		if (disabled_conn_id >= 0) {	// it is a former connection
			if (Random_Double (0.0f, 1.0f, true, false) < reactivateConnectionThresh) {
				connections [disabled_conn_id].enabled = true;	// former connection is reactivated
				return true;
			} else {
				logger->warn ("process ended well but no connection has been added during Genome<Args...>::AddConnection");
				return true;	// return true even no connection has been change because process ended well
			}
		} else {
			// innovId
			const unsigned int innov_id = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);

			// weight
			const double weight = Random_Double (- weightExtremumInit, weightExtremumInit);

			connections.push_back (Connection (innov_id, inNodeId, outNodeId, inNodeRecu, weight, true));

			// update layers
			if (inNodeRecu == 0) {
				// the added connection may have an impact on the network
				if (nodes [inNodeId]->layer >= nodes [outNodeId]->layer) {
					// it has an impact
					nodes [outNodeId]->layer = nodes [inNodeId]->layer + 1;	// the node is constraint, its layer is the following one
					UpdateLayers (outNodeId);
				}
			}

			return true;
		}
	} else {
		logger->warn ("maximum iteration threshold to find a valid connection has been reached in Genome<Args...>::AddConnection: no connection is added");
		return false;	// cannot find a valid connection
	}
}

template <typename... Args>
bool Genome<Args...>::AddMonotypedNode (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxIterationsFindConnectionThresh) {
	logger->trace ("adding a node");
	// choose at random an enabled connection
	if (connections.size() > 0) {	// if there is no connection, we cannot add a node!
		unsigned int iConn = Random_UInt (0, (unsigned int) connections.size () - 1);
		unsigned int iterationNb = 0;
		while (iterationNb < maxIterationsFindConnectionThresh && !connections [iConn].enabled) {
			iConn = Random_UInt (0, (unsigned int) connections.size () - 1);
			iterationNb ++;
		}
		if (iterationNb < maxIterationsFindConnectionThresh) {	// a connection has been found
			// disable former connection
			connections [iConn].enabled = false;
			
			// setup new node
			const unsigned int newNodeId = (unsigned int) nodes.size ();
			const unsigned int iT_in = nodes [connections [iConn].inNodeId]->index_T_in;
			const unsigned int iT_out = nodes [connections [iConn].outNodeId]->index_T_out;

			// get Node<T_in, T_out>
			nodes.push_back(CreateNode::get<Args...> (iT_in, iT_out));

			// setup the node
			nodes.back ()->id = newNodeId;
			nodes.back ()->layer = -1;	// no layer for now
			nodes.back ()->index_T_in = iT_in;
			nodes.back ()->index_T_out = iT_out;
			nodes.back ()->index_activation_fn = Random_UInt (0, (unsigned int) activationFns [iT_in][iT_out].size () - 1);
			nodes.back ()->setActivationFn (
				activationFns [iT_in][iT_out][nodes.back ()->index_activation_fn]->clone (false)	//activation function with new fresh parameters
			);
			nodes.back ()->innovId = node_innov->getInnovId (
				nodes.back ()->index_T_in,
				nodes.back ()->index_T_out,
				nodes.back ()->index_activation_fn,
				RepetitionNodeCheck (nodes.back ()->index_T_in, nodes.back ()->index_T_out, nodes.back ()->index_activation_fn) - 1
			);
			nodes.back ()->setResetValue (resetValues [iT_in]);

			// build first connection
			int inNodeId = connections [iConn].inNodeId;
			int outNodeId = newNodeId;
			unsigned int inNodeRecu = connections [iConn].inNodeRecu;
			unsigned int innovId = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);
			double weight = connections [iConn].weight;

			connections.push_back (Connection (innovId, inNodeId, outNodeId, inNodeRecu, weight, true));
			
			// build second connection
			inNodeId = newNodeId;
			outNodeId = connections [iConn].outNodeId;
			inNodeRecu = 0;
			innovId = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);
			weight = Random_Double (- weightExtremumInit, weightExtremumInit);

			connections.push_back (Connection (innovId, inNodeId, outNodeId, inNodeRecu, weight, true));

			// update layers
			if (connections [iConn].inNodeRecu > 0) {	// the connection was recurrent, so the layers are not changed
				if (nodes [connections [iConn].outNodeId]->layer  == 1) {	// the output node was on the first layer, we cannot set the new node on the layer 0 (reserved for the input): everything has to moved
					nodes [newNodeId]->layer = 1;	// update newNodeId layer
					nodes [connections [iConn].outNodeId]->layer = 2;	// update outNodeId layer
					UpdateLayers (connections [iConn].outNodeId);	// update other layers
				} else {
					// we set the new node's layer to the first one because there is no node connected to it on the same network (with a null recurrency) 
					nodes [newNodeId]->layer = 1;	// update newNodeId layer
				}
			} else {									// else, the node is one layer further in the network
				nodes [newNodeId]->layer = nodes [connections [iConn].inNodeId]->layer + 1;	// update newNodeId layer
				if (nodes [newNodeId]->layer >= nodes [connections [iConn].outNodeId]->layer) {
					nodes [connections [iConn].outNodeId]->layer = nodes [newNodeId]->layer + 1;	// the node is constraint, its layer is the following one
					UpdateLayers (connections [iConn].outNodeId);
				}
			}
			return true;
		} else {
			logger->warn ("maximum iteration threshold to find an active connection has been reached in Genome<Args...>::AddMonotypedNode: no node is added");
			return false;	// no active connection found
		}
	} else {
		logger->warn ("there is no connection, no node is added in Genome<Args...>::AddMonotypedNode");
		return false;	// there is no connection, cannot add a node
	}
}

template <typename... Args>
bool Genome<Args...>::AddBitypedNode (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindNodeThresh) {
	logger->trace ("adding a bi-typed node");
	if (N_types > 1) {	// if there is only one type, we cannot add a bi-typed node!
		// Add bi-typed node
		const unsigned int newNodeId = (unsigned int) nodes.size ();
		const unsigned int iT_in = Random_UInt (0, N_types - 1);
		unsigned int iT_out = Random_UInt (0, N_types - 1);
		while (iT_out == iT_in) {
			iT_out = Random_UInt (0, N_types - 1);
		}

		// get Node<T_in, T_out>
		nodes.push_back(CreateNode::get<Args...> (iT_in, iT_out));

		// setup the node
		nodes.back ()->id = newNodeId;
		nodes.back ()->layer = 1;	// default to first layer
		nodes.back ()->index_T_in = iT_in;
		nodes.back ()->index_T_out = iT_out;
		nodes.back ()->index_activation_fn = Random_UInt (0, (unsigned int) activationFns [iT_in][iT_out].size () - 1);
		nodes.back ()->setActivationFn (
			activationFns [iT_in][iT_out][nodes.back ()->index_activation_fn]->clone (false)	//activation function with new fresh parameters
		);
		nodes.back ()->innovId = node_innov->getInnovId (
			nodes.back ()->index_T_in,
			nodes.back ()->index_T_out,
			nodes.back ()->index_activation_fn,
			RepetitionNodeCheck (nodes.back ()->index_T_in, nodes.back ()->index_T_out, nodes.back ()->index_activation_fn) - 1
		);
		nodes.back ()->setResetValue (resetValues [iT_in]);

		// Add the first connection
		unsigned int inNodeId = Random_UInt (0, (unsigned int) nodes.size () - 1);
		unsigned int inNodeRecu = Random_UInt (0, maxRecurrency);
		unsigned int iterationNb = 0;
		while (
			iterationNb < maxIterationsFindNodeThresh
			&& (
				nodes [inNodeId]->index_T_out != iT_in
				|| (
					inNodeId >= nbBias + nbInput && inNodeId < nbBias + nbInput + nbOutput	// cannot build a non recurrent connection with an output node as the input's connection
					&& inNodeRecu == 0
				)
			)
		) {
			inNodeId = Random_UInt (0, (unsigned int) nodes.size () - 1);
			inNodeRecu = Random_UInt (0, maxRecurrency);
			iterationNb ++;
		}
		if (iterationNb == maxIterationsFindNodeThresh) {
			logger->warn ("maximum iteration threshold to find a valid connection has been reached in Genome<Args...>::AddBitypedNode: a bi-typed node has been added, but no connection point or start to it");
			return false;
		}

		unsigned int innov_id = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [newNodeId]->innovId, inNodeRecu);
		double weight = Random_Double (- weightExtremumInit, weightExtremumInit);

		connections.push_back(Connection (innov_id, inNodeId, newNodeId, inNodeRecu, weight, true));

		// update newNode's layer
		if (inNodeRecu > 0) {
			nodes [newNodeId]->layer = 1;	// the node is not constraint as the only input connection comes from a reccurent node
		} else {
			nodes [newNodeId]->layer = nodes [inNodeId]->layer + 1;	// the node is constraint, its layer is the following one
		}

		// Add the second connection
		unsigned int outNodeId = Random_UInt (0, (unsigned int) nodes.size () - 1);
		inNodeRecu = 0;
		iterationNb = 0;
		while (
			iterationNb < maxIterationsFindNodeThresh
			&& !(CheckNewConnectionValidity (newNodeId, outNodeId, inNodeRecu))
		) {
			outNodeId = Random_UInt (0, (unsigned int) nodes.size () - 1);
			iterationNb ++;
		}
		if (iterationNb == maxIterationsFindNodeThresh) {
			logger->warn ("maximum iteration threshold to find a valid connection has been reached in Genome<Args...>::AddBitypedNode: a bi-typed node has been added, but only one connection point or start to it");
			return false;
		}

		innov_id = conn_innov->getInnovId (nodes [newNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);
		weight = Random_Double (- weightExtremumInit, weightExtremumInit);

		connections.push_back(Connection (innov_id, newNodeId, outNodeId, inNodeRecu, weight, true));

		// update layers
		if (nodes [newNodeId]->layer >= nodes [outNodeId]->layer) {
			nodes [outNodeId]->layer = nodes [newNodeId]->layer + 1;	// the node is constraint, its layer is the following one
			UpdateLayers (outNodeId);
		}

		return true;
	} else {
		logger->warn ("the genome is processing one type of object: cannot add a bi-typed node in Genome<Args...>::AddBitypedNode");
		return false;	// there is only one type of object
	}
}

template <typename... Args>
void Genome<Args...>::UpdateLayers_Recursive (unsigned int nodeId) {
	for (size_t iConn = 0; iConn < connections.size (); iConn ++) {
		if (
			!(connections [iConn].inNodeRecu > 0)
			&& connections [iConn].enabled
			&& connections [iConn].inNodeId == nodeId
		) {
			unsigned int newNodeId = connections [iConn].outNodeId;

			if (nodes [newNodeId]->layer <= nodes [nodeId]->layer) {
				nodes [newNodeId]->layer = nodes [nodeId]->layer + 1;

				UpdateLayers_Recursive (newNodeId);
			}
		}
	}
}

template <typename... Args>
void Genome<Args...>::UpdateLayers (int nodeId) {
	// Update layers
	UpdateLayers_Recursive (nodeId);

	// this might move some output's node, let's homogenize that
	int outputLayer = nodes [nbBias + nbInput]->layer;
	for (size_t i = nbBias + nbInput; i < nbBias + nbInput + nbOutput; i ++) {
		// check among the outputs which one is the highest and set the output layer to it
		if (nodes [i]->layer > outputLayer) {
			outputLayer = nodes [i]->layer;
		}
	}
	for (size_t i = nbBias + nbInput + nbOutput; i < nodes.size (); i ++) {
		// check among the hiddens which one is the highest and set the output layer to it + 1
		if (nodes [i]->layer >= outputLayer) {
			outputLayer = nodes [i]->layer + 1;
		}
	}
	for (size_t i = nbBias + nbInput; i < nbBias + nbInput + nbOutput; i ++) {
		// new layer!
		nodes [i]->layer = outputLayer;
	}
}

template <typename... Args>
std::unique_ptr<Genome<Args...>> Genome<Args...>::clone () {
	std::unique_ptr<Genome<Args...>> genome =  std::make_unique<Genome<Args...>> (nbBias, nbInput, nbOutput, N_types, resetValues, activationFns, weightExtremumInit, logger);

	genome->nodes.reserve (nodes.size ());
	for (const std::unique_ptr<NodeBase>& node : nodes) {
		genome->nodes.push_back (node->clone ());
	}
	genome->connections = connections;
	genome->speciesId = speciesId;
	genome->fitness = fitness;

	return genome;
}


template <typename... Args>
void Genome<Args...>::print (const std::string& prefix) {
	std::cout << prefix << "Number of Bias Node: " << nbBias << std::endl;
	std::cout << prefix << "Number of Input Node: " << nbInput << std::endl;
	std::cout << prefix << "Number of Output Node: " << nbOutput << std::endl;
	std::cout << prefix << "Weight's range at intialization: [" << -1.0f * weightExtremumInit << ", " << weightExtremumInit << "]" << std::endl;
	std::cout << prefix << "Number of objects manipulated: " << N_types << std::endl;
	std::cout << prefix << "Current Fitness: " << fitness << std::endl;
	std::cout << prefix << "Current SpeciesID: " << speciesId << std::endl;
	std::cout << prefix << "Number of Activation Functions [Input TypeID to Output TypeID (Number of functions)]: ";
	for (size_t i = 0; i < activationFns.size (); i++) {
		for (size_t j = 0; j < activationFns [i].size (); j++) {
			std::cout << i << " to " << j << " (" << activationFns [i][j].size () << "), ";
		}
	}
	std::cout << std::endl;
	std::cout << prefix << "Nodes: " << std::endl;
	for (const std::unique_ptr<NodeBase>& node : nodes) {
		node->print (prefix + "   ");
		std::cout << std::endl;
	}
	std::cout << prefix << "Connections: " << std::endl;
	for (size_t i = 0; i < connections.size (); i++) {
		connections [i].print (prefix + "   ");
		std::cout << std::endl;
	}
}

template <typename... Args>
void Genome<Args...>::draw (const std::string& font_path, unsigned int windowWidth, unsigned int windowHeight, float dotsRadius) {
	sf::RenderWindow window (sf::VideoMode (windowWidth, windowHeight), "PNEATM - https://github.com/titofra");

    std::vector<sf::CircleShape> dots;
	std::vector<sf::Text> dotsText;
	std::vector<sf::VertexArray> lines;
	sf::Text mainText;

    // ### NODES ###
	sf::Font font;
	if (!font.loadFromFile(font_path)) {
		std::cout << "Error while loading font in 'Genome<Args...>::draw'." << std::endl;
	}

	for (size_t i = 0; i < nodes.size (); i++) {
		dots.push_back (sf::CircleShape ());
		dotsText.push_back (sf::Text ());

		dots [i].setRadius (dotsRadius);
		dots [i].setFillColor (sf::Color::White);

		dotsText [i].setString (std::to_string (i));
		dotsText [i].setFillColor (sf::Color::White);
		dotsText [i].setCharacterSize (20);
		dotsText [i].setFont (font);
	}

	const unsigned int nbLayer = nodes [nbBias + nbInput]->layer + 1;

	// constants for position x
	const float firstLayerX = 175.0f;
	const float stepX = 0.9f * ((float) windowWidth - firstLayerX) / (float) (nbLayer - 1);

	// input
	if (nbBias + nbInput == 1) {	// if there is only one node, we draw it on the middle of y	// Note that this not recommended to do a network without any Bias node
		dots [0].setPosition ({firstLayerX - dotsRadius, 0.5f * (float) windowHeight  - dotsRadius});
		dotsText [0].setPosition ({firstLayerX - dotsRadius, 0.5f * (float) windowHeight + 4.0f});
	} else {
		for (unsigned int i = 0; i < nbBias + nbInput; i++) {
			dots [i].setPosition ({firstLayerX - dotsRadius, 0.1f * (float) windowHeight + (float) i * 0.8f * (float) windowHeight / (float) (nbBias + nbInput - 1) - dotsRadius});
			dotsText [i].setPosition ({firstLayerX - dotsRadius, 0.1f * (float) windowHeight + (float) i * 0.8f * (float) windowHeight / (float) (nbBias + nbInput - 1) + 4.0f});
		}
	}
	// output
	if (nbOutput == 1) {	// if there is only one node, we draw it on the middle of y
		dots [nbBias + nbInput].setPosition ({firstLayerX + stepX * (float) nodes[nbBias + nbInput]->layer - dotsRadius, 0.5f * (float) windowHeight  - dotsRadius});
		dotsText [nbBias + nbInput].setPosition ({firstLayerX + stepX * (float) nodes[nbBias + nbInput]->layer - dotsRadius, 0.5f * (float) windowHeight + 4.0f});
	} else {
		for (unsigned int i = nbBias + nbInput; i < nbBias + nbInput + nbOutput; i++) {
			dots [i].setPosition ({firstLayerX + stepX * (float) nodes[i]->layer - dotsRadius, 0.1f * (float) windowHeight + (float) (i - (nbBias + nbInput)) * 0.8f * (float) windowHeight / (float) (nbOutput - 1) - dotsRadius});
			dotsText [i].setPosition ({firstLayerX + stepX * (float) nodes[i]->layer - dotsRadius, 0.1f * (float) windowHeight + (float) (i - (nbBias + nbInput)) * 0.8f * (float) windowHeight / (float) (nbOutput - 1) + 4.0f});
		}
	}
	// other
	for (unsigned int ilayer = 1; ilayer < nbLayer - 1; ilayer++) {
		std::vector<unsigned int> iNodesiLayer;
		for (unsigned int i = nbBias + nbInput + nbOutput; i < (unsigned int) nodes.size(); i++) {
			if (nodes [i]->layer == (int) ilayer) {
				iNodesiLayer.push_back (i);
			}
		}
		if (iNodesiLayer.size() == 1) {	// if there is only one node, we draw it on the middle of y
			dots [iNodesiLayer [0]].setPosition ({firstLayerX + stepX * (float) nodes[iNodesiLayer[0]]->layer - dotsRadius, 0.5f * (float) windowHeight - dotsRadius});
			dotsText [iNodesiLayer [0]].setPosition ({firstLayerX + stepX * (float) nodes[iNodesiLayer[0]]->layer - dotsRadius, 0.5f * (float) windowHeight + 4.0f});
		} else {
			for (size_t i = 0; i < iNodesiLayer.size(); i++) {
				dots [iNodesiLayer [i]].setPosition ({firstLayerX + stepX * (float) nodes[iNodesiLayer[i]]->layer - dotsRadius, 0.1f * (float) windowHeight + (float) i * 0.8f * (float) windowHeight / (float) (iNodesiLayer.size() - 1) - dotsRadius});
				dotsText [iNodesiLayer [i]].setPosition ({firstLayerX + stepX * (float) nodes[iNodesiLayer[i]]->layer - dotsRadius, 0.1f * (float) windowHeight + (float) i * 0.8f * (float) windowHeight / (float) (iNodesiLayer.size() - 1) + 4.0f});
			}
		}
	}
	
	// ### CONNECTIONS ###
	double maxWeight = connections[0].weight;
	for (size_t i = 1; i < connections.size(); i++) {
		if (connections [i].weight * connections [i].weight > maxWeight * maxWeight) {
			maxWeight = connections [i].weight;
		}
	}

	for (size_t i = 0; i < connections.size(); i++) {
		sf::Color color;
		if (connections [i].enabled) {
			if (!(connections [i].inNodeRecu > 0)) {
				color = sf::Color::Green;
			} else {
				color = sf::Color::Blue;
			}
		} else {
			if (!(connections [i].inNodeRecu > 0)) {
				color = sf::Color::Red;
			} else {
				color = sf::Color::Yellow;
			}
		}

		// weighted connections
		if (connections [i].weight / maxWeight > 0.0) {
			float ratioColor = (float) pow(connections[i].weight / maxWeight, 0.4);
			color.r = static_cast<sf::Uint8>(color.r * ratioColor);
			color.g = static_cast<sf::Uint8>(color.g * ratioColor);
			color.b = static_cast<sf::Uint8>(color.b * ratioColor);
		} else {
			float ratioColor = (float) pow(-1 * connections[i].weight / maxWeight, 0.4);
			color.r = static_cast<sf::Uint8>(color.r * ratioColor);
			color.g = static_cast<sf::Uint8>(color.g * ratioColor);
			color.b = static_cast<sf::Uint8>(color.b * ratioColor);
		}

		lines.push_back (sf::VertexArray (sf::Lines, 2));
		lines [i][0] = sf::Vertex({dots [connections [i].inNodeId].getPosition ().x + dotsRadius, dots [connections [i].inNodeId].getPosition ().y + dotsRadius}, color);
		lines [i][1] = sf::Vertex({dots [connections [i].outNodeId].getPosition ().x + dotsRadius, dots [connections [i].outNodeId].getPosition ().y + dotsRadius}, color);
	}

	// ### TEXT ###
	mainText.setFillColor (sf::Color::White);
	mainText.setCharacterSize (11);
	mainText.setFont (font);
	mainText.setPosition ({15.0, 15.0});

	sf::String stringMainText = "";
	for (size_t i = 0; i < connections.size(); i++) {
		stringMainText += std::to_string (connections [i].inNodeId) + "  ->  " + std::to_string (connections [i].outNodeId) + "   (" +  std::to_string (connections [i].weight) + ")";
		if (connections [i].inNodeRecu > 0) {
			stringMainText += " R (";
			stringMainText += std::to_string (connections [i].inNodeRecu);
			stringMainText += ")";
		}
		if (!connections [i].enabled) {
			stringMainText += " D";
		}
		stringMainText += "\n";
	}
	mainText.setString (stringMainText);

    while (window.isOpen ()) {
        sf::Event event;
        while (window.pollEvent (event))
        {
            if (event.type == sf::Event::Closed) {
                window.close ();
            }
        }

        window.clear(sf::Color::Black);

		for (size_t i = 0; i < connections.size(); i++) {
			window.draw (lines [i]);
		}
        for (size_t i = 0; i < nodes.size(); i++) {
		    window.draw (dots [i]);
		    window.draw (dotsText [i]);
		}
		window.draw (mainText);
        window.display ();
    }
}

template <typename... Args>
void Genome<Args...>::serialize (std::ofstream& outFile) {
	Serialize (nbBias, outFile);
	Serialize (nbInput, outFile);
	Serialize (nbOutput, outFile);
	Serialize (weightExtremumInit, outFile);
	Serialize (N_types, outFile);
	Serialize (nbBias, outFile);

	Serialize (nodes.size (), outFile);
	for (size_t k = 0; k < nodes.size (); k++) {
		Serialize (nodes [k]->index_T_in, outFile);
		Serialize (nodes [k]->index_T_out, outFile);
		nodes [k]->serialize (outFile);
	}

	Serialize (connections.size (), outFile);
	for (Connection& conn : connections) {
		conn.serialize (outFile);
	}

	Serialize (fitness, outFile);
	Serialize (speciesId, outFile);
	Serialize (N_runNetwork, outFile);
}

template <typename... Args>
void Genome<Args...>:: deserialize (std::ifstream& inFile) {
	Deserialize (nbBias, inFile);
	Deserialize (nbInput, inFile);
	Deserialize (nbOutput, inFile);
	Deserialize (weightExtremumInit, inFile);
	Deserialize (N_types, inFile);
	Deserialize (nbBias, inFile);

	size_t sz;

	Deserialize (sz, inFile);
	nodes.clear ();
	nodes.reserve (sz);
	for (size_t k = 0; k < sz; k++) {
		unsigned int iT_in, iT_out;
		Deserialize (iT_in, inFile);
		Deserialize (iT_out, inFile);
		nodes.push_back (CreateNode::get<Args...> (iT_in, iT_out));
		nodes [k]->deserialize (inFile, activationFns);
	}

	Deserialize (sz, inFile);
	connections.clear ();
	connections.reserve (sz);
	for (size_t k = 0; k < sz; k++) {
		connections.push_back (Connection (inFile));
	}

	Deserialize (fitness, inFile);
	Deserialize (speciesId, inFile);
	Deserialize (N_runNetwork, inFile);
}


#endif	// GENOME_HPP