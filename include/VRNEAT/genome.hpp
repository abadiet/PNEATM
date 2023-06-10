#ifndef GENOME_HPP
#define GENOME_HPP

#include <VRNEAT/Node/node_base.hpp>
#include <VRNEAT/Node/node.hpp>
#include <VRNEAT/Connection/connection.hpp>
#include <VRNEAT/Connection/innovation.hpp>
#include <VRNEAT/utils.hpp>
#include <vector>
#include <SFML/Graphics.hpp>
#include <cmath>
#include <functional>

namespace vrneat {

template <typename... Args>
class Genome {
	public:
		Genome (std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_init, std::vector<void*> resetValues, std::vector<std::vector<std::vector<std::function <void* (void*)>>>> activationFns, innovation_t* conn_innov, unsigned int N_ConnInit, float probRecuInit, float weightExtremumInit, unsigned int maxRecuInit);

		template <typename T_in>
		void loadInputs (T_in inputs []);
		template <typename T_in>
		void loadInput (T_in input, int input_id);

		void runNetwork ();

		template <typename T_out>
		void getOutputs (T_out outputs []);
		template <typename T_out>
		T_out getOutput (int output_id);

		void mutate (innovation_t* conn_innov, unsigned int maxRecurrency = 0, float mutateWeightThresh = 0.8f, float mutateWeightFullChangeThresh = 0.1f, float mutateWeightFactor = 1.2f, float addConnectionThresh = 0.05f, int maxIterationsFindConnectionThresh = 20, float reactivateConnectionThresh = 0.25f, float addNodeThresh = 0.03f, int maxIterationsFindNodeThresh = 20, float addTranstypeThresh = 0.02f);

		//void drawNetwork (sf::Vector2u windowSize = {1300, 800}, float dotsRadius = 6.5f);

	private:
		unsigned int nbBias;
		unsigned int nbInput;
		unsigned int nbOutput;
		float weightExtremumInit;
		unsigned int N_types;
		std::vector<std::vector<std::vector<std::function <void* (void*)>>>> activationFns;
		std::vector<void*> resetValues;

		std::vector <NodeBase*> nodes;
		std::vector <std::vector <NodeBase*>> prevNodes;
		unsigned int rec_max;
		std::vector <Connection> connections;

		float fitness;
		int speciesId;

		bool CheckNewConnectionValidity (unsigned int inNodeId, unsigned int outNodeId, unsigned int inNodeRecu, unsigned int* disabled_conn_id = nullptr);
		bool CheckNewConnectionCircle (unsigned int inNodeId, unsigned int outNodeId);
		void MutateWeights (float mutateWeightFullChangeThresh, float mutateWeightFactor);
		bool AddConnection (innovation_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindConnectionThresh, float reactivateConnectionThresh);
		bool AddNode (innovation_t* conn_innov, unsigned int maxIterationsFindNodeThresh);
		bool AddTranstype (innovation_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindNodeThresh);
		void UpdateLayers (int inNodeId);
		void UpdateLayers_Recursive (int inNodeId);
};

}

#endif	// GENOME_HPP