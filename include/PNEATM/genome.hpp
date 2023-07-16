#ifndef GENOME_HPP
#define GENOME_HPP

#include <PNEATM/Node/node_base.hpp>
#include <PNEATM/Node/innovation_node.hpp>
#include <PNEATM/Connection/connection.hpp>
#include <PNEATM/Connection/innovation_connection.hpp>
#include <PNEATM/utils.hpp>
#include <vector>
#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <cstring>
#include <spdlog/spdlog.h>
#include <memory>


/* HEADER */

namespace pneatm {

typedef struct mutationParams {
	struct Nodes {
		float rate = 0.03f;
		float monotypedRate = 0.5f;
		struct Monotyped {
			unsigned int maxIterationsFindConnection = 20;
		};
		struct Monotyped monotyped;
		struct Bityped {
			unsigned int maxRecurrencyEntryConnection;
			unsigned int maxIterationsFindNode = 20;
		};
		struct Bityped bityped;
	};
	struct Nodes nodes;
	struct Connections {
		float rate = 0.05f;
		float reactivateRate = 0.25f;
		unsigned int maxRecurrency;
		unsigned int maxIterations = 20;
		unsigned int maxIterationsFindNode;
	};
	struct Connections connections;
	struct Weights {
		float rate = 0.05f;
		float fullChangeRate = 0.1f;
		float perturbationFactor = 1.2f;
	};
	struct Weights weights;
} mutationParams_t;

template <typename... Args>
class Genome {
	public:
		Genome (std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_init, std::vector<void*> resetValues, std::vector<std::vector<std::vector<void*>>> activationFns, innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int N_ConnInit, float probRecuInit, float weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger);
		Genome (unsigned int nbBias, unsigned int nbInput, unsigned int nbOutput, unsigned int N_types, std::vector<void*> resetValues, std::vector<std::vector<std::vector<void*>>> activationFns, float weightExtremumInit, spdlog::logger* logger);
		~Genome ();

		float getFitness () {return fitness;};
		int getSpeciesId () {return speciesId;};

		template <typename T_in>
		void loadInputs (std::vector<T_in> inputs);
		template <typename T_in>
		void loadInput (T_in input, int input_id);

		void runNetwork ();

		template <typename T_out>
		std::vector<T_out> getOutputs ();
		template <typename T_out>
		T_out getOutput (int output_id);

		void mutate (innovationConn_t* conn_innov, innovationNode_t* node_innov, mutationParams_t params);

		std::unique_ptr<Genome<Args...>> clone ();

		void print (std::string prefix = "");
		void draw (std::string font_path, unsigned int windowWidth = 1300, unsigned int windowHeight = 800, float dotsRadius = 6.5f);

	private:
		unsigned int nbBias;
		unsigned int nbInput;
		unsigned int nbOutput;
		float weightExtremumInit;
		unsigned int N_types;
		std::vector<std::vector<std::vector<void*>>> activationFns;
		std::vector<void*> resetValues;

		std::vector <std::unique_ptr<NodeBase>> nodes;
		std::vector <std::vector <void*>> prevNodes;	//TODO: void* -> const void*
		std::vector <Connection> connections;

		float fitness;
		int speciesId;

		spdlog::logger* logger;

		unsigned int RepetitionNodeCheck (unsigned int index_T_in, unsigned int index_T_out, unsigned int index_activation_fn);
		bool CheckNewConnectionValidity (unsigned int inNodeId, unsigned int outNodeId, unsigned int inNodeRecu, int* disabled_conn_id = nullptr);
		bool CheckNewConnectionCircle (unsigned int inNodeId, unsigned int outNodeId);
		void MutateWeights (float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor);
		bool AddConnection (innovationConn_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindConnectionThresh, float reactivateConnectionThresh);
		bool AddNode (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxIterationsFindConnectionThresh);
		bool AddTranstype (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindNodeThresh);
		void UpdateLayers (int nodeId);
		void UpdateLayers_Recursive (unsigned int nodeId);

	template <typename... Args2>
	friend class Population;
};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename... Args>
Genome<Args...>::Genome (std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_init, std::vector<void*> resetValues, std::vector<std::vector<std::vector<void*>>> activationFns, innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int N_ConnInit, float probRecuInit, float weightExtremumInit, unsigned int maxRecuInit, spdlog::logger* logger) :
	weightExtremumInit (weightExtremumInit),
	activationFns (activationFns),
	resetValues (resetValues),
	logger (logger)
{
	logger->trace ("Genome initialization");

	N_types = (unsigned int) activationFns.size ();
	speciesId = -1;

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
			nodes.back ()->innovId = node_innov->getInnovId (
				nodes.back ()->index_T_in,
				nodes.back ()->index_T_out,
				nodes.back ()->index_activation_fn,
				RepetitionNodeCheck (nodes.back ()->index_T_in, nodes.back ()->index_T_out, nodes.back ()->index_activation_fn) - 1
			);
			nodes.back ()->loadInput (bias_init [i]);	// load input now as it will always be the same
			nodes.back ()->process ();	// load output now as it will always be the same
			// bias node doesn't requires a reset value as they are never changed

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
					activationFns [i][j][nodes.back ()->index_activation_fn]
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
		if (Random_Float (0.0f, 1.0f, true, false) < probRecuInit) {
			inNodeRecu = Random_UInt (0, maxRecuInit);
		}
		if (CheckNewConnectionValidity (inNodeId, outNodeId, inNodeRecu)) {	// we don't care of former connections as there is no disabled connection for now
			// innovId
			const unsigned int innov_id = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);

			// weight
			const float weight = Random_Float (- weightExtremumInit, weightExtremumInit);

			connections.push_back(Connection (innov_id, inNodeId, outNodeId, inNodeRecu, weight, true));

			if (inNodeRecu == 0 && nodes [outNodeId]->layer <= nodes [inNodeId]->layer) {
				nodes [outNodeId]->layer = nodes [inNodeId]->layer + 1;
				UpdateLayers (outNodeId);
			}

			iConn ++;
		}
	}
}

template <typename... Args>
Genome<Args...>::Genome (unsigned int nbBias, unsigned int nbInput, unsigned int nbOutput, unsigned int N_types, std::vector<void*> resetValues, std::vector<std::vector<std::vector<void*>>> activationFns, float weightExtremumInit, spdlog::logger* logger) :
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
}

template <typename... Args>
Genome<Args...>::~Genome () {
	logger->trace ("Genome destruction");
}

template <typename... Args>
template <typename T_in>
void Genome<Args...>::loadInputs (std::vector<T_in> inputs) {
	for (unsigned int i = 0; i < nbInput; i++) {
		nodes [i + nbBias]->loadInput (static_cast<void*> (&inputs [i]));
	}
}

template <typename... Args>
template <typename T_in>
void Genome<Args...>::loadInput (T_in input, int input_id) {
	nodes [input_id + nbBias]->loadInput (static_cast<void*> (&input));
}

template <typename... Args>
void Genome<Args...>::runNetwork() {
	/* Process all input and output. For that, it "scans" each layer from the inputs to the last hidden's layer to calculate input with already known value. */ 

	// reset input
	for (size_t i = nbBias + nbInput; i < nodes.size(); i++) {
		nodes [i]->reset ();
	}

	// process nodes[*]->output for input/bias nodes
	for (unsigned int i = 0; i < nbBias + nbInput; i++) {
		nodes [i]->process ();
	}

	int lastLayer = nodes[nbBias + nbInput]->layer;

	for (int ilayer = 1; ilayer <= lastLayer; ilayer++) {
		// process nodes[*]->input
		for (size_t i = 0; i < connections.size (); i++) {
			if (connections [i].enabled && nodes [connections [i].outNodeId]->layer == ilayer) {	// if the connections still exist and is pointing on the current layer
				if (connections [i].inNodeRecu == 0) {
					logger->trace ("processing of connection {3}: adding to node{0}'s input value from dot product between node{1}'s output and {2}", connections [i].outNodeId, connections [i].inNodeId, connections [i].weight, i);
					nodes [connections [i].outNodeId]->AddToInput (
						nodes [connections [i].inNodeId]->getOutput (),
						connections [i].weight
					);
				} else {	// is recurent
					if (connections[i].inNodeRecu < (unsigned int) prevNodes.size () + 1) {
						logger->trace ("processing of connection {3}: adding to node{0}'s input value from dot product between node{1}'s output ({4} generation from now) and {2}", connections [i].outNodeId, connections [i].inNodeId, connections [i].weight, i, connections [i].inNodeRecu);
						nodes [connections[i].outNodeId]->AddToInput (
							prevNodes [(unsigned int) prevNodes.size () - connections [i].inNodeRecu][connections [i].inNodeId],
							connections [i].weight
						);
					} else {
						logger->trace ("processing of connection {3}: cannot add to node{0}'s input value from dot product between node{1}'s output ({4} generation from now) and {2} as this value isn't existing yet!", connections [i].outNodeId, connections [i].inNodeId, connections [i].weight, i, connections [i].inNodeRecu);
						// the input of the connection isn't existing yet!
						// we consider that the connection isn't existing
					}
				}
			}
		}

		// process nodes[*]->output
		for (size_t i = 0; i < nodes.size (); i++) {
			if (nodes [i]->layer == ilayer) {
				nodes [i]->process ();
			}
		}
	}

	// we have to store previous values
	logger->trace ("saving previous nodes' outputs");
	prevNodes.push_back ({});
	prevNodes.back ().reserve (nodes.size ());
	for (const std::unique_ptr<NodeBase>& node : nodes) {
		prevNodes.back ().push_back (node->getOutput ());
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
void Genome<Args...>::mutate(innovationConn_t* conn_innov, innovationNode_t* node_innov, mutationParams_t params) {
	// WEIGHTS
	MutateWeights (params.weights.rate, params.weights.fullChangeRate, params.weights.perturbationFactor);

	// NODES
	if (Random_Float (0.0f, 1.0f, true, false) < params.nodes.rate) {
		if (Random_Float (0.0f, 1.0f, true, false) < params.nodes.monotypedRate) {
			AddNode (conn_innov, node_innov, params.nodes.monotyped.maxIterationsFindConnection);
		} else {
			AddTranstype (conn_innov, node_innov, params.nodes.bityped.maxRecurrencyEntryConnection, params.nodes.bityped.maxIterationsFindNode);
		}
	}

	// CONNECTIONS
	if (Random_Float (0.0f, 1.0f, true, false) < params.connections.rate) {
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
void Genome<Args...>::MutateWeights (float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor) {
	logger->trace ("mutating of weights");
	for (size_t i = 0; i < connections.size (); i++) {
		if (Random_Float (0.0f, 1.0f, true, false) < mutateWeightThresh) {
			if (Random_Float (0.0f, 1.0f, true, false) < mutateWeightFullChangeThresh) {
				// reset weight
				logger->trace ("resetting connection{}'s weight", i);
				connections [i].weight = Random_Float (- weightExtremumInit, weightExtremumInit);
			} else {
				// pertub weight
				logger->trace ("pertubating connection{}'s weight", i);
				connections [i].weight *= Random_Float (- mutateWeightFactor, mutateWeightFactor);
			}
		}
	}
}

template <typename... Args>
bool Genome<Args...>::AddConnection (innovationConn_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindConnectionThresh, float reactivateConnectionThresh) {	// return true if the process ended well, false in the other case
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
		// mutating
		if (disabled_conn_id >= 0) {	// it is a former connection
			if (Random_Float (0.0f, 1.0f, true, false) < reactivateConnectionThresh) {
				logger->trace ("connection{} is re-enabled", disabled_conn_id);
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
			const float weight = Random_Float (- weightExtremumInit, weightExtremumInit);

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

			logger->trace ("connection{0} added between node{1} (with a recurrency of {3}) and node{2}", connections.size () - 1, inNodeId, outNodeId, inNodeRecu);
			return true;
		}
	} else {
		logger->warn ("maximum iteration threshold to find a valid connectino has been reached in Genome<Args...>::AddConnection: no connection is added");
		return false;	// cannot find a valid connection
	}
}

template <typename... Args>
bool Genome<Args...>::AddNode (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxIterationsFindConnectionThresh) {	// return true = node created, false = nothing created
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

			logger->trace ("adding node{2}, linking type{0} to type{1}", iT_in, iT_out, newNodeId);

			// get Node<T_in, T_out>
			nodes.push_back(CreateNode::get<Args...> (iT_in, iT_out));

			// setup the node
			nodes.back ()->id = newNodeId;
			nodes.back ()->layer = -1;	// no layer for now
			nodes.back ()->index_T_in = iT_in;
			nodes.back ()->index_T_out = iT_out;
			nodes.back ()->index_activation_fn = Random_UInt (0, (unsigned int) activationFns [iT_in][iT_out].size () - 1);
			nodes.back ()->setActivationFn (
				activationFns [iT_in][iT_out][nodes.back ()->index_activation_fn]
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
			float weight = connections [iConn].weight;

			logger->trace ("adding connection{0} between node{1} and node{2}", connections.size (), inNodeId, outNodeId);

			connections.push_back (Connection (innovId, inNodeId, outNodeId, inNodeRecu, weight, true));
			
			// build second connection
			inNodeId = newNodeId;
			outNodeId = connections [iConn].outNodeId;
			inNodeRecu = 0;
			innovId = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);
			weight = Random_Float (- weightExtremumInit, weightExtremumInit);

			logger->trace ("adding connection{0} between node{1} and node{2}", connections.size (), inNodeId, outNodeId);

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
			logger->warn ("maximum iteration threshold to find an active connection has been reached in Genome<Args...>::AddNode: no node is added");
			return false;	// no active connection found
		}
	} else {
		logger->warn ("there is no connection, no node is added in Genome<Args...>::AddNode");
		return false;	// there is no connection, cannot add a node
	}
}

template <typename... Args>
bool Genome<Args...>::AddTranstype (innovationConn_t* conn_innov, innovationNode_t* node_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindNodeThresh) {
	logger->trace ("adding a bi-typed node");
	if (N_types > 1) {	// if there is only one type, we cannot add a bi-typed node!
		// Add bi-typed node
		const unsigned int newNodeId = (unsigned int) nodes.size ();
		const unsigned int iT_in = Random_UInt (0, N_types - 1);
		unsigned int iT_out = Random_UInt (0, N_types - 1);
		while (iT_out == iT_in) {
			iT_out = Random_UInt (0, N_types - 1);
		}

		logger->trace ("adding node{2}, linking type{0} to type{1}", iT_in, iT_out, newNodeId);

		// get Node<T_in, T_out>
		nodes.push_back(CreateNode::get<Args...> (iT_in, iT_out));

		// setup the node
		nodes.back ()->id = newNodeId;
		nodes.back ()->layer = 1;	// default to first layer
		nodes.back ()->index_T_in = iT_in;
		nodes.back ()->index_T_out = iT_out;
		nodes.back ()->index_activation_fn = Random_UInt (0, (unsigned int) activationFns [iT_in][iT_out].size () - 1);
		nodes.back ()->setActivationFn (
			activationFns [iT_in][iT_out][nodes.back ()->index_activation_fn]
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
			logger->warn ("maximum iteration threshold to find a valid connection has been reached in Genome<Args...>::AddTranstype: a bi-typed node has been added, but no connection point or start to it");
			return false;
		}

		unsigned int innov_id = conn_innov->getInnovId (nodes [inNodeId]->innovId, nodes [newNodeId]->innovId, inNodeRecu);
		float weight = Random_Float (- weightExtremumInit, weightExtremumInit);

		logger->trace ("adding connection{0} between node{1} and node{2}", connections.size (), inNodeId, newNodeId);

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
			logger->warn ("maximum iteration threshold to find a valid connection has been reached in Genome<Args...>::AddTranstype: a bi-typed node has been added, but only one connection point or start to it");
			return false;
		}

		innov_id = conn_innov->getInnovId (nodes [newNodeId]->innovId, nodes [outNodeId]->innovId, inNodeRecu);
		weight = Random_Float (- weightExtremumInit, weightExtremumInit);

		logger->trace ("adding connection{0} between node{1} and node{2}", connections.size (), newNodeId, outNodeId);

		connections.push_back(Connection (innov_id, newNodeId, outNodeId, inNodeRecu, weight, true));

		// update layers
		if (nodes [newNodeId]->layer >= nodes [outNodeId]->layer) {
			nodes [outNodeId]->layer = nodes [newNodeId]->layer + 1;	// the node is constraint, its layer is the following one
			UpdateLayers (outNodeId);
		}

		return true;
	} else {
		logger->warn ("the genome is processing one type of object: cannot add a bi-typed node in Genome<Args...>::AddTranstype");
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
void Genome<Args...>::print (std::string prefix) {
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
	std::cout << prefix << "Previous Nodes: " << prevNodes.size () << " calls to the network are stored" << std::endl;
	std::cout << prefix << "Connections: " << std::endl;
	for (size_t i = 0; i < connections.size (); i++) {
		connections [i].print (prefix + "   ");
		std::cout << std::endl;
	}
}


template <typename... Args>
void Genome<Args...>::draw (std::string font_path, unsigned int windowWidth, unsigned int windowHeight, float dotsRadius) {
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
	float maxWeight = connections[0].weight;
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

#endif	// GENOME_HPP