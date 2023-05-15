#ifndef GENOME_HPP
#define GENOME_HPP

#include <LRNEAT/node.hpp>
#include <LRNEAT/connection.hpp>
#include <vector>
#include <SFML/Graphics.hpp>

namespace neat {

class Genome{
	public:
		Genome (std::vector<int> nbInput, std::vector<int> nbOutput, std::vector<int> nbHiddenInit, float probConnInit, std::vector<std::vector<int>>* innovIds, int* lastInnovId, float weightExtremumInit = 20.0f);
		void loadInputs (void* inputs []);
		void loadInput (void* input, int input_id);
		void runNetwork (std::vector<ActivationFn> activationFns);
		void getOutputs (void* outputs []);
		void getOutput (void* output, int output_id);
		void mutate (std::vector<std::vector<int>>* innovIds, int* lastInnovId, bool areRecurrentConnectionsAllowed = false, float mutateWeightThresh = 0.8f, float mutateWeightFullChangeThresh = 0.1f, float mutateWeightFactor = 1.2f, float addConnectionThresh = 0.05f, int maxIterationsFindConnectionThresh = 20, float reactivateConnectionThresh = 0.25f, float addNodeThresh = 0.03f, int maxIterationsFindNodeThresh = 20);
		void drawNetwork (sf::Vector2u windowSize = {1300, 800}, float dotsRadius = 6.5f);

	private:
		int nbInput;
		int nbOutput;
		float fitness;
		int speciesId;
		
		std::vector<std::vector<Node>> nodes;
		std::vector<std::vector<Connection>> connections;

		float weightExtremumInit;
	
		int getInnovId(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int inNodeId, int outNodeId);
		void mutateWeights(float mutateWeightFullChangeThresh, float mutateWeightFactor);
		bool addConnection(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindConnectionThresh, bool areRecurrentConnectionsAllowed, float reactivateConnectionThresh);
		int isValidNewConnection(int inNodeId, int outNodeId, bool areRecurrentConnectionsAllowed);
		bool addNode(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindNodeThresh, bool areRecurrentConnectionsAllowed);
		void updateLayersRec(int nodeId);
};

}

#endif	// GENOME_HPP