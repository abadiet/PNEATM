#include <NEAT/genome.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace neat;

Genome::Genome(int nbInput, int nbOutput, int nbHiddenInit, float probConnInit, std::vector<std::vector<int>>* innovIds, int* lastInnovId, float weightExtremumInit): weightExtremumInit(weightExtremumInit), nbInput(nbInput), nbOutput(nbOutput){
	speciesId = -1;
	// NODES
	// bias
	nodes.push_back(Node(0, 0));
	nodes[0].sumInput = 1;	// init value of the bias node
	// input
	for (int i = 1; i < nbInput + 1; i++) {
		nodes.push_back(Node(i, 0));
	}
	// output
	int outputLayer;
	if (nbHiddenInit > 0) {
		outputLayer = 2;
	} else {
		outputLayer = 1;
	}
	for (int i = nbInput + 1; i < nbInput + 1 + nbOutput; i++) {
		nodes.push_back(Node(i, outputLayer));
	}
	// hidden
	for (int i = nbInput + 1 + nbOutput; i < nbInput + 1 + nbOutput + nbHiddenInit; i++) {
		nodes.push_back(Node(i, 1));
	}
	
	// CONNECTIONS
	// input -> hidden
	for (int inNodeId = 0; inNodeId < nbInput + 1; inNodeId++) {	// input
		for (int outNodeId = nbInput + 1 + nbOutput; outNodeId < nbInput + 1 + nbOutput + nbHiddenInit; outNodeId++) {	// hidden
			if ((float) rand() / (float) RAND_MAX <= probConnInit) {
				int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
				connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			}
		}
	}
	// hidden -> output
	for (int inNodeId = nbInput + 1 + nbOutput; inNodeId < nbInput + 1 + nbOutput + nbHiddenInit; inNodeId++) {	// hidden
		for (int outNodeId = nbInput + 1; outNodeId < nbInput + 1 + nbOutput; outNodeId++) {	// output
			if ((float) rand() / (float) RAND_MAX <= probConnInit) {
				int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
				connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			}
		}
	}
	// input -> output
	for (int inNodeId = 0; inNodeId < nbInput + 1; inNodeId++) {	// input
		for (int outNodeId = nbInput + 1; outNodeId < nbInput + 1 + nbOutput; outNodeId++) {	// output
			if ((float) rand() / (float) RAND_MAX <= probConnInit) {
				int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
				connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			}
		}
	}
}

int Genome::getInnovId(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int inNodeId, int outNodeId) {
	/* get the innovation id of the connection inNodeId -> outNodeId. Create one if needed */
	if ((int) innovIds->size() < inNodeId + 1) {
		int previousSize = (int) innovIds->size();
		for (int k = 0; k < inNodeId + 1 - previousSize; k++) {	// we complete the array until we reach inNodeId
			innovIds->push_back({-1});
		}
	}
	if ((int) (*innovIds)[inNodeId].size() < outNodeId + 1) {
		int previousSize = (int) (*innovIds)[inNodeId].size();
		for (int k = 0; k < outNodeId + 1 - previousSize; k++) {	// we complete the array until we reach outNodeId
			(*innovIds)[inNodeId].push_back(-1);
		}
	}
	if ((*innovIds)[inNodeId][outNodeId] == -1) {	// has been this connection build in the past ?
		*lastInnovId += 1;
		(*innovIds)[inNodeId][outNodeId] = *lastInnovId;
	}
	
	return (*innovIds)[inNodeId][outNodeId];
}


void Genome::loadInputs(float inputs[]) {
	for (int i = 0; i < nbInput; i++) {
		nodes[i + 1].sumInput = inputs[i];	// i + 1 because the first node is the bias one
		nodes[i + 1].sumOutput = inputs[i];	// sumInput = sumOutput for the inputs nodes
	}
}


void Genome::runNetwork(float activationFn(float input)) {
	/* Process all sumInput and sumOutput. For that, it "scans" each layer from the inputs to the last hidden's layer to calculate sumInput with already known value. */ 
	
	// reset sumInput
	for (int i = nbInput + 1; i < (int) nodes.size(); i++) {
		nodes[i].sumInput = 0;
	}
	
	int lastLayer = nodes[1 + nbInput].layer;
	
	for (int ilayer = 0; ilayer < lastLayer; ilayer++) {
		// process sumInput
		for (int i = 0; i < (int) connections.size(); i++) {
			if (connections[i].enabled) {	// if the connections still exist
				if (nodes[connections[i].inNodeId].layer == ilayer) {
					if (nodes[connections[i].inNodeId].layer < nodes[connections[i].outNodeId].layer) {	// if the connection is normally oriented, else we already used it
						nodes[connections[i].outNodeId].sumInput += nodes[connections[i].inNodeId].sumOutput * connections[i].weight;
					}
				} else {
					if (nodes[connections[i].outNodeId].layer == ilayer) {
						if (nodes[connections[i].outNodeId].layer < nodes[connections[i].inNodeId].layer) {	// if the connection is anormally oriented, else we already used it
							nodes[connections[i].inNodeId].sumInput += nodes[connections[i].outNodeId].sumOutput * connections[i].weight;
						}
					}
				}
			}
		}
		
		// process sumOutput
		for (int i = 1 + nbInput; i < (int) nodes.size(); i++) {	// for each node except inputs because sumOutput has already be calculated
			if (nodes[i].layer == ilayer + 1) {
				nodes[i].sumOutput = activationFn(nodes[i].sumInput);
			}
		}
	}
}


void Genome::getOutputs(float outputs[]) {
	for (int i = 0; i < nbOutput; i++) {
		outputs[i] = nodes[1 + nbInput + i].sumOutput;
	}
}


void Genome::mutate(std::vector<std::vector<int>>* innovIds, int* lastInnovId, bool areRecurrentConnectionsAllowed, float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor, float addConnectionThresh, int maxIterationsFindConnectionThresh, float reactivateConnectionThresh, float addNodeThresh, int maxIterationsFindNodeThresh) {
	// ### WEIGHTS ###
	float randomNb = (float) rand() / (float) RAND_MAX;
	while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
		randomNb = (float) rand() / (float) RAND_MAX;
	}	// generate a random value in [0,1)
	if (randomNb < mutateWeightThresh) {
		// mutating weights
		mutateWeights(mutateWeightFullChangeThresh, mutateWeightFactor);
	}
	
	// ### CONNECTIONS ###
	randomNb = (float) rand() / (float) RAND_MAX;
	while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
		randomNb = (float) rand() / (float) RAND_MAX;
	}	// generate a random value in [0,1)
	if (randomNb < addConnectionThresh) {
		// adding a conection
		addConnection(innovIds, lastInnovId, maxIterationsFindConnectionThresh, areRecurrentConnectionsAllowed, reactivateConnectionThresh);
	}
	
	// ### NODES ###
	randomNb = (float) rand() / (float) RAND_MAX;
	while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
		randomNb = (float) rand() / (float) RAND_MAX;
	}	// generate a random value in [0,1)
	if (randomNb < addNodeThresh) {
		// adding a node
		addNode(innovIds, lastInnovId, maxIterationsFindNodeThresh, areRecurrentConnectionsAllowed);
	}
}

void Genome::mutateWeights(float mutateWeightFullChangeThresh, float mutateWeightFactor) {
	for (int i = 0; i < (int) connections.size(); i++) {
		float randomNb = (float) rand() / (float) RAND_MAX;
		while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
			randomNb = (float) rand() / (float) RAND_MAX;
		}	// generate a random value in [0,1)
		if (randomNb < mutateWeightFullChangeThresh) {
			// reset weight
			connections[i].weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;
		} else {
			// pertub weight
			connections[i].weight *= (float) rand() / (float) RAND_MAX * 2 * mutateWeightFactor - mutateWeightFactor;
		}
	}
}

bool Genome::addConnection(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindConnectionThresh, bool areRecurrentConnectionsAllowed, float reactivateConnectionThresh) {	// return true if the process ended well, false in the other case
	// find valid node pair
	int iterationNb = 0;
	int isValid = 0;
	int inNodeId = rand() % (int) nodes.size();
	int outNodeId = rand() % (int) nodes.size();
	while (iterationNb < maxIterationsFindConnectionThresh && isValid == 0) {
		inNodeId = rand() % (int) nodes.size();
		outNodeId = rand() % (int) nodes.size();
		isValid = isValidNewConnection(inNodeId, outNodeId, areRecurrentConnectionsAllowed);
		iterationNb++;
	}
	
	if (iterationNb < maxIterationsFindConnectionThresh) {	// a valid connection has been found
		// mutating
		if (isValid == 2) {	// it is a former connection
			float randomNb = (float) rand() / (float) RAND_MAX;
			while ((int) randomNb == 1) {
				randomNb = (float) rand() / (float) RAND_MAX;
			}	// generate a random value in [0,1)
			if (randomNb < reactivateConnectionThresh) {
				// search the connection
				int iConnDisabled = 0;
				while (iConnDisabled < (int) connections.size() && !(!connections[iConnDisabled].enabled && connections[iConnDisabled].inNodeId == inNodeId && connections[iConnDisabled].outNodeId == outNodeId)) {
					iConnDisabled ++;
				}
				if (iConnDisabled < (int) connections.size()) {
					connections[iConnDisabled].enabled = true;	// former connection is reactivaded
					return true;
				} else {
					std::cout << "Error : Genome::addConnection" << std::endl;	// impossible
					return false;
				}
			} else {
				return true;	// return true even no connection has been change because process ended well
			}
		} else {
			int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
			float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
			bool recurrent = false;
			if (isValid == 3) {
				recurrent = true;
			}
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, recurrent));
			return true;
		}
	} else {
		return false;	// cannot find a valid connection
	}
}

int Genome::isValidNewConnection(int inNodeId, int outNodeId, bool areRecurrentConnectionsAllowed) {	// 0 = not valid connection, 1 = valid connection, 2 = connection currently disabled, 3 = recurrent connections
	if (inNodeId == outNodeId) return 0;	// the connection boucle itself
	if (nodes[inNodeId].layer == nodes[outNodeId].layer) return 0;	// the connection link to nodes on the same layer
	for (int i = 0; i < (int) connections.size(); i++) {
		if (connections[i].inNodeId == inNodeId && connections[i].outNodeId == outNodeId) {
			if (connections[i].enabled) {
				return 0;	// it is already a connection enabled
			} else {
				return 2;	// it is a former connection
			}
		}
	}
	if (nodes[inNodeId].layer > nodes[outNodeId].layer)	{
		if (areRecurrentConnectionsAllowed) {
			return 3;	// recurrent connections are allowed
		} else {
			return 0;	// recurrent connection are disallowed
		}
	}
	return 1;	// test done : it is a valid connection !
}

bool Genome::addNode(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindNodeThresh, bool areRecurrentConnectionsAllowed) {	// return true = node created, false = nothing created
	// choose at random an enabled forward connection
	if ((int) connections.size() > 0) {
		int iConn = rand() % (int) connections.size();
		int iterationNb = 0;
		while (iterationNb < maxIterationsFindNodeThresh && !(connections[iConn].enabled && nodes[connections[iConn].inNodeId].layer < nodes[connections[iConn].outNodeId].layer)) {
			iConn = rand() % (int) connections.size();
			iterationNb ++;
		}
		if (iterationNb < maxIterationsFindNodeThresh) {
			// disable former connection
			connections[iConn].enabled = false;
			
			// setup new node
			int newNodeId = (int) nodes.size();
			nodes.push_back(Node(newNodeId, -1));	// no layer for the moment
			
			// build first connection
			int inNodeId = connections[iConn].inNodeId;
			int outNodeId = newNodeId;
			int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
			float weight = connections[iConn].weight;
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			
			// build second connection
			inNodeId = newNodeId;
			outNodeId = connections[iConn].outNodeId;
			innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
			weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			
			// update layers
			nodes[newNodeId].layer = nodes[connections[iConn].inNodeId].layer + 1;	// update newNodeId layer
			nodes[connections[iConn].outNodeId].layer = nodes[newNodeId].layer + 1;	// update outNodeId layer
			updateLayersRec(connections[iConn].outNodeId);	// recursively update layers
			
			// output nodes can have different nodes after updating layers: let's give output nodes the same output layer, the maximum one
			int maxLayer = nodes[1 + nbInput + nbOutput].layer;
			for (int i = 1 + nbInput + nbOutput + 1; i < (int) nodes.size(); i++) {
				if (nodes[i].layer > maxLayer) {
					maxLayer = nodes[i].layer;
				}
			}
			for (int i = 1 + nbInput; i < 1 + nbInput + nbOutput; i++) {
				nodes[i].layer = maxLayer + 1;
			}
			
			// recurrent connections can have moved during the process and can now be disabled or non-recurent: check Neat Ai's youtube video to understand the phenomenon ("Neat Ai does XOR Mutate" at timecode 5:40)
			if (areRecurrentConnectionsAllowed) {	// if potentially there is recurrent connections
				for (int i = 0; i < (int) connections.size(); i++) {
					if (connections[i].isRecurrent) {
						if (connections[i].enabled && nodes[connections[i].inNodeId].layer == nodes[connections[i].outNodeId].layer) {	// tthe connection is now illegal and whould be disabled
							connections[i].enabled = false;
						} else {
							if (nodes[connections[i].inNodeId].layer < nodes[connections[i].outNodeId].layer) {	// the connection became non-recurrent
								connections[i].isRecurrent = false;
							}
						}
					}
				}
			}
			
			return true;
		} else {
			std::cout << "Error : no active connection found in Genome::addNode" << std::endl;
			return false;	// no active connection found
		}
	} else {
		return false;	// there is no connection, cannot add a node
	}
}

void Genome::updateLayersRec(int nodeId) {
	for (int iConn = 0; iConn < (int) connections.size(); iConn++) {
		if (!connections[iConn].isRecurrent && connections[iConn].enabled && connections[iConn].inNodeId == nodeId) {
			int newNodeId = connections[iConn].outNodeId;
			nodes[newNodeId].layer = nodes[nodeId].layer + 1;
			updateLayersRec(newNodeId);
		}
	}
}

void Genome::drawNetwork(sf::Vector2u windowSize, float dotsRadius) {
	sf::RenderWindow window(sf::VideoMode(windowSize.x, windowSize.y), "NEAT - Titofra");
    
    sf::CircleShape dots[nodes.size()];
	sf::Text dotsText[nodes.size()];
	sf::Vertex lines[connections.size()][2];
	sf::Text mainText;
    
    // ### NODES ###
	sf::Font font;
	if (!font.loadFromFile("/usr/share/fonts/cantarell/Cantarell-VF.otf")) {
		std::cout << "Error while loading font in 'Genome::drawNetwork'." << std::endl;
	}

	for (int i = 0; i < (int) nodes.size(); i++) {
		dots[i].setRadius(dotsRadius);
		dots[i].setFillColor(sf::Color::White);
		
		dotsText[i].setString(std::to_string(i));	// need <iostream> for std::to_string function
		dotsText[i].setFillColor(sf::Color::White);
		dotsText[i].setCharacterSize(20);
		dotsText[i].setFont(font);
	}

	// NOTE: the indice of the first output node is 1 + nbInput and the indice of the first hidden node is 1 + nbInput + nbOutput

	int nbLayer = nodes[1 + nbInput].layer + 1;

	// variables for position x
	float firstLayerX = 200;
	float stepX = (float) (0.9 * windowSize.x - firstLayerX) / (float) (nbLayer - 1);

	// input
	for (int i = 0; i < 1 + nbInput; i++) {
		dots[i].setPosition({(float) (firstLayerX + stepX * (float) nodes[i].layer - (float) dotsRadius), (float) (0.1 * windowSize.y + i * 0.8 * windowSize.y / nbInput - (float) dotsRadius)});
		dotsText[i].setPosition({(float) (firstLayerX + stepX * (float) nodes[i].layer - dotsRadius), (float) (0.1 * windowSize.y + i * 0.8 * windowSize.y / nbInput + 4.0)});
	}
	// output
	if (nbOutput == 1) {	// if there is only one node, we draw it on the middle of y
		dots[1 + nbInput].setPosition({(float) (firstLayerX + stepX * (float) nodes[1 + nbInput].layer - dotsRadius), (float) (0.5 * windowSize.y  - dotsRadius)});
		dotsText[1 + nbInput].setPosition({(float) (firstLayerX + stepX * (float) nodes[1 + nbInput].layer - dotsRadius), (float) (0.5 * windowSize.y + 4.0)});
	} else {
		for (int i = 1 + nbInput; i < 1 + nbInput + nbOutput; i++) {
			dots[i].setPosition({(float) (firstLayerX + stepX * (float) nodes[i].layer - dotsRadius), (float) (0.1 * windowSize.y + (i - (1 + nbInput)) * 0.8 * windowSize.y / (nbOutput - 1) - dotsRadius)});
			dotsText[i].setPosition({(float) (firstLayerX + stepX * (float) nodes[i].layer - dotsRadius), (float) (0.1 * windowSize.y + (i - (1 + nbInput)) * 0.8 * windowSize.y / (nbOutput - 1) + 4.0)});
		}
	}
	// other
	for (int ilayer = 1; ilayer < (nbLayer - 1); ilayer++) {
		std::vector<int> iNodesiLayer;
		for (int i = 1 + nbInput + nbOutput; i < (int) nodes.size(); i++) {
			if (nodes[i].layer == ilayer) {
				iNodesiLayer.push_back(i);
			}
		}
		if ((int) iNodesiLayer.size() == 1) {	// if there is only one node, we draw it on the middle of y
			dots[iNodesiLayer[0]].setPosition({(float) (firstLayerX + stepX * (float) nodes[iNodesiLayer[0]].layer - dotsRadius), (float) (0.5 * windowSize.y - dotsRadius)});
			dotsText[iNodesiLayer[0]].setPosition({(float) (firstLayerX + stepX * (float) nodes[iNodesiLayer[0]].layer - dotsRadius), (float) (0.5 * windowSize.y + 4.0)});
		} else {
			for (int i = 0; i < (int) iNodesiLayer.size(); i++) {
				dots[iNodesiLayer[i]].setPosition({(float) (firstLayerX + stepX * (float) nodes[iNodesiLayer[i]].layer - dotsRadius), (float) (0.1 * windowSize.y + i * 0.8 * windowSize.y / ((int) iNodesiLayer.size() - 1) - dotsRadius)});
				dotsText[iNodesiLayer[i]].setPosition({(float) (firstLayerX + stepX * (float) nodes[iNodesiLayer[i]].layer - dotsRadius), (float) (0.1 * windowSize.y + i * 0.8 * windowSize.y / ((int) iNodesiLayer.size() - 1) + 4.0)});
			}
		}
	}
	
	// ### CONNECTIONS ###
	float maxWeight = connections[0].weight;
	for (int i = 1; i < (int) connections.size(); i++) {
		if (connections[i].weight * connections[i].weight > maxWeight * maxWeight) {	// square to only compare positive value
			maxWeight = connections[i].weight;
		}
	}

	for (int i = 0; i < (int) connections.size(); i++) {
		sf::Color color;
		if (connections[i].enabled) {
			if (!connections[i].isRecurrent) {
				color = sf::Color::Green;
			} else {
				color = sf::Color::Blue;
			}
		} else {
			color = sf::Color::Red;
		}

		// weighted connections
		if (connections[i].weight / maxWeight > 0.0) {
			float ratioColor = (float) pow(connections[i].weight / maxWeight, 0.3);
			color.r = static_cast<sf::Uint8>(color.r * ratioColor);
			color.g = static_cast<sf::Uint8>(color.g * ratioColor);
			color.b = static_cast<sf::Uint8>(color.b * ratioColor);
		} else {
			if (connections[i].weight < 1e-10) {	// == 0
				color = sf::Color::Black;	// Warnings: works because bakground is black !
			} else {
				float ratioColor = (float) pow(-1 * connections[i].weight / maxWeight, 0.4);
				color.r = static_cast<sf::Uint8>(color.r * ratioColor);
				color.g = static_cast<sf::Uint8>(color.g * ratioColor);
				color.b = static_cast<sf::Uint8>(color.b * ratioColor);
			}
		}


		lines[i][0] = sf::Vertex({(float) dots[connections[i].inNodeId].getPosition().x + dotsRadius, (float) dots[connections[i].inNodeId].getPosition().y + dotsRadius}, color);
		lines[i][1] = sf::Vertex({(float) dots[connections[i].outNodeId].getPosition().x + dotsRadius, (float) dots[connections[i].outNodeId].getPosition().y + dotsRadius}, color);
	}

	// ### TEXT ###
	mainText.setFillColor(sf::Color::White);
	mainText.setCharacterSize(15);
	mainText.setFont(font);
	mainText.setPosition({15.0, 15.0});

	sf::String stringMainText = "";
	for (int i = 0; i < (int) connections.size(); i++) {
		stringMainText += std::to_string(connections[i].inNodeId) + "  <->  " + std::to_string(connections[i].outNodeId) + "   (" +  std::to_string(connections[i].weight) + ")";	// need <iostream> for std::to_string function
		if (connections[i].enabled && connections[i].isRecurrent) {
			stringMainText += " R";
		} else {
			if (!connections[i].enabled) {
				stringMainText += " D";
			}
		}
		stringMainText += "\n";
	}
	mainText.setString(stringMainText);
    
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        window.clear(sf::Color::Black);

		for (int i = 0; i < (int) connections.size(); i++) {
			window.draw(lines[i], 2, sf::PrimitiveType::Lines);
		}
        for (int i = 0; i < (int) nodes.size(); i++) {
		    window.draw(dots[i]);
		    window.draw(dotsText[i]);
		}
		window.draw(mainText);
        window.display();
    }
}
