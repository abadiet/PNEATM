#ifndef GENOME_HPP
#define GENOME_HPP

#include <VRNEAT/node.hpp>
#include <VRNEAT/connection.hpp>
#include <VRNEAT/utils.hpp>
#include <VRNEAT/activation_fn.hpp>
#include <vector>
#include <SFML/Graphics.hpp>

namespace vrneat {

class Genome {
	public:
		Genome (std::vector<int> bias_sch, std::vector<int> inputs_sch, std::vector<int> outputs_sch, std::vector<int> hiddens_sch_init, std::vector<void*> bias_init, float probConnInit, std::vector<std::vector<int>>* innovIds, int* lastInnovId, float weightExtremumInit = 20.0f);
		void loadInputs (void* inputs []);
		void loadInput (void* input, int input_id);
		void runNetwork (std::vector<ActivationFn> activationFns);
		void getOutputs (void* outputs []);
		void* getOutput (int output_id);
		void mutate (std::vector<std::vector<int>>* innovIds, int* lastInnovId, bool areRecurrentConnectionsAllowed = false, float mutateWeightThresh = 0.8f, float mutateWeightFullChangeThresh = 0.1f, float mutateWeightFactor = 1.2f, float addConnectionThresh = 0.05f, int maxIterationsFindConnectionThresh = 20, float reactivateConnectionThresh = 0.25f, float addNodeThresh = 0.03f, int maxIterationsFindNodeThresh = 20);
		void drawNetwork (sf::Vector2u windowSize = {1300, 800}, float dotsRadius = 6.5f);

	private:
		int nbBias;
		int nbInput;
		int nbOutput;
		std::vector<Node> nodes;
		std::vector<std::vector<void*>> prevOut;
		float weightExtremumInit;

		float fitness;
		int speciesId;
		std::vector<Connection> connections;
	
		int findFuncId (int in_kind, int out_kind);
		int getInnovId(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int inNodeId, int outNodeId);
		void mutateWeights(float mutateWeightFullChangeThresh, float mutateWeightFactor);
		bool addConnection(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindConnectionThresh, bool areRecurrentConnectionsAllowed, float reactivateConnectionThresh);
		int isValidNewConnection(int inNodeId, int outNodeId, bool areRecurrentConnectionsAllowed);
		bool addNode(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindNodeThresh, bool areRecurrentConnectionsAllowed);
		void updateLayersRec(int nodeId);
};

}

#endif	// GENOME_HPP