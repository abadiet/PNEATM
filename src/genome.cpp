#include <VRNEAT/genome.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace vrneat;

Genome::Genome (std::vector<int> bias_sch, std::vector<int> inputs_sch, std::vector<int> outputs_sch, std::vector<int> hiddens_sch_init, std::vector<void*> bias_init, float probConnInit, std::vector<std::vector<int>>* innovIds, int* lastInnovId, float weightExtremumInit): weightExtremumInit(weightExtremumInit) {
	speciesId = -1;
	// NODES
	// bias
	nbBias = 0;
	for (int i = 0; i < (int) bias_sch.size (); i++) {
		for (int k = 0; k < bias_sch [i]; k++) {
			int func_id = findFuncId (i, i);
			nodes.push_back(Node(nbBias, 0, i, i, func_id));
			nodes.back ().input = bias_init [nbBias];	// init value of the bias node
			nbBias ++;
		}
	}
	// input
	nbInput = 0;
	for (int i = 0; i < (int) input_sch.size (); i++) {
		for (int k = 0; k < input_sch [nbBias + nbInput]; k++) {
			int func_id = findFuncId (i, i);
			nodes.push_back(Node(nbBias + nbInput, 0, i, i, func_id));
			nbInput ++;
		}
	}
	// output
	nbOutput = 0;
	int outputLayer;
	if ((int) nbHiddenInit.size () > 0) {
		outputLayer = 2;
	} else {
		outputLayer = 1;
	}
	for (int i = 0; i < (int) outputs_sch.size (); i++) {
		for (int k = 0; k < outputs_sch [nbOutput]; k++) {
			int func_id = findFuncId (i, i);
			nodes.push_back(Node(nbBias + nbInput + nbOutput, outputLayer, i, i, func_id));
			nbOutput ++;
		}
	}
	// hidden
	int nbHidden = 0;
	for (int i = 0; i < (int) hiddens_sch_init.size (); i++) {
		for (int k = 0; k < hiddens_sch_init [nbHidden]; k++) {
			int func_id = findFuncId (i, i);
			nodes.push_back(Node(nbBias + nbInput + nbOutput + nbHidden, 1, i, i, func_id));
			nbHidden ++;
		}
	}
	
	// CONNECTIONS
	// input -> hidden
	for (int inNodeId = 0; inNodeId < nbBias + nbInput; inNodeId++) {	// input
		for (int outNodeId = nbBias + nbInput + nbOutput; outNodeId < nbBias + nbInput  + nbOutput + nbHiddenInit; outNodeId++) {	// hidden
			if ((float) rand() / (float) RAND_MAX <= probConnInit) {
				int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
				connections.push_back(Connection(innovId, inNodeId, outNodeId, 0, weight, true, false));
			}
		}
	}
	// hidden -> output
	for (int inNodeId = nbBias + nbInput + nbOutput; inNodeId < nbBias + nbInput + nbOutput + nbHiddenInit; inNodeId++) {	// hidden
		for (int outNodeId = nbBias + nbInput; outNodeId < nbBias + nbInput + nbOutput; outNodeId++) {	// output
			if ((float) rand() / (float) RAND_MAX <= probConnInit) {
				int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
				connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			}
		}
	}
	// input -> output
	for (int inNodeId = 0; inNodeId < nbBias + nbInput; inNodeId++) {	// input
		for (int outNodeId = nbBias + nbInput; outNodeId < nbBias + nbInput + nbOutput; outNodeId++) {	// output
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
	if ((*innovIds)[inNodeId][outNodeId] == -1) {	// has this connection been built in the past ?
		*lastInnovId ++;
		(*innovIds)[inNodeId][outNodeId] = *lastInnovId;
	}
	
	return (*innovIds)[inNodeId][outNodeId];
}


void Genome::loadInputs(void* inputs []) {
	for (int i = 0; i < nbInput; i++) {
		nodes[i + 1].input = inputs[i];	// i + 1 because the first node is the bias one
		nodes[i + 1].output = inputs[i];	// input = output for the inputs nodes
	}
}

void Genome::loadInput(void* input, int input_id) {
	nodes[input_id + 1].input = input;	// i + 1 because the first node is the bias one
	nodes[input_id + 1].output = input;	// input = output for the inputs nodes
}

void Genome::runNetwork(std::vector<ActivationFn> activationFns) {
	/* Process all input and output. For that, it "scans" each layer from the inputs to the last hidden's layer to calculate input with already known value. */ 

	// reset input
	for (int i = nbInput + 1; i < (int) nodes.size(); i++) {
		nodes[i].input = 0;
	}
	
	int lastLayer = nodes[nbBias + nbInput].layer;
	
	for (int ilayer = 1; ilayer < lastLayer; ilayer++) {
		// process nodes[*].input
		for (int i = 0; i < (int) connections.size(); i++) {
			if (connections[i].enabled && nodes[connections[i].outNodeId].layer == ilayer) {	// if the connections still exist and is pointing on the current layer
				if (connections[i].inNodeRecu == 0) {	// no recurence
					nodes[connections[i].outNodeId].input += nodes[connections[i].inNodeId].output * connections[i].weight;
				} else {	// is recurent
					if (connections[i].inNodeRecu <= (int) prevOut.size ()) {
						nodes[connections[i].outNodeId].input += prevOut[(int) prevOut.size () - connections[i].inNodeRecu][connections[i].inNodeId] * connections[i].weight;
					} else {
						// the input of the connection isn't existing yet!
						// we consider that the connection isn't existing
					}
				}
			}
		}
		
		// process nodes[*].output
		for (int i = nbBias + nbInput; i < (int) nodes.size(); i++) {	// for each node except inputs because output has already be calculated
			if (nodes[i].layer == ilayer) {
				nodes[i].output = activationFns[nodes[i].func_id](nodes[i].input);
			}
		}
	}
}

void Genome::getOutputs(void* outputs[]) {
	for (int i = 0; i < nbOutput; i++) {
		outputs[i] = nodes[nbBias + nbInput + i].output;
	}
}

void* Genome::getOutput (int output_id) {
	return nodes[nbBias + nbInput + output_id].output;
}

void Genome::mutate(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxRecurrency, float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor, float addConnectionThresh, int maxIterationsFindConnectionThresh, float reactivateConnectionThresh, float addNodeThresh, int maxIterationsFindNodeThresh) {
	// ### WEIGHTS ###
	if (Random_Float (0.0f, 1.0f, true, false) < mutateWeightThresh) {
		// mutating weights
		mutateWeights(mutateWeightFullChangeThresh, mutateWeightFactor);
	}
	
	// ### NODES ###
	if (Random_Float (0.0f, 1.0f, true, false) < addNodeThresh) {
		// adding a node
		addNode(innovIds, lastInnovId, maxIterationsFindNodeThresh, maxRecurrency);
	}

	// ### CONNECTIONS ###
	if (Random_Float (0.0f, 1.0f, true, false) < addConnectionThresh) {
		// adding a conection
		addConnection(innovIds, lastInnovId, maxIterationsFindConnectionThresh, maxRecurrency, reactivateConnectionThresh);
	}
}

void Genome::mutateWeights(float mutateWeightFullChangeThresh, float mutateWeightFactor) {
	for (int i = 0; i < (int) connections.size(); i++) {
		if (Random_Float (0.0f, 1.0f, true, false) < mutateWeightFullChangeThresh) {
			// reset weight
			connections[i].weight = Random_Float (- weightExtremumInit, weightExtremumInit);
		} else {
			// pertub weight
			connections[i].weight *= Random_Float (- mutateWeightFactor, mutateWeightFactor)
		}
	}
}

bool Genome::addConnection(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindConnectionThresh, bool areRecurrentConnectionsAllowed, float reactivateConnectionThresh) {	// return true if the process ended well, false in the other case
	// find valid node pair
	int iterationNb = 0;
	int inNodeId, outNodeId, inNodeRecu;
	int isValid = 0;
	int disabled_conn_id = -1;
	while (iterationNb < maxIterationsFindConnectionThresh && isValid) {
		inNodeId = rand() % (int) nodes.size();
		outNodeId = rand() % (int) nodes.size();
		inNodeRecu = rand() % (maxRecurrency + 1);
		isValid = isValidNewConnection(inNodeId, outNodeId, inNodeId, &disabled_conn_id);
		iterationNb++;
	}
	
	if (iterationNb < maxIterationsFindConnectionThresh) {	// a valid connection has been found
		// mutating
		if (disabled_conn_id >= 0) {	// it is a former connection
			if (Random_Float (0.0f, 1.0f, true, false) < reactivateConnectionThresh) {
				connections[disabled_conn_id].enabled = true;	// former connection is reactivated
				return true;
			} else {
				return true;	// return true even no connection has been change because process ended well
			}
		} else {
			int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId, inNodeRecu);
			float weight = Random_Float (- weightExtremumInit, weightExtremumInit);	// random number in [-weightExtremumInit; weightExtremumInit]
			connections.push_back(Connection(innovId, inNodeId, outNodeId, inNodeRecu, weight, true));
			return true;
		}
	} else {
		return false;	// cannot find a valid connection
	}
}

bool Genome::isValidNewConnection(int inNodeId, int outNodeId, int inNodeRecu, int* disabled_conn_id) {
	if (inNodeId == outNodeId) return false;	// the connection boucle itself
	for (int i = 0; i < (int) connections.size(); i++) {
		if (
			connections[i].inNodeId == inNodeId
			&& connections[i].outNodeId == outNodeId
			&& connections[i].inNodeRecu == inNodeRecu
		) {
			if (connections[i].enabled) {
				return false;	// it is already an enabled connection
			} else {
				*disabled_conn_id = i;
				return true;	// it is a former connection
			}
		}
	}
	return true;	// tests passed well: it is a valid connection!
}

bool Genome::addNode(std::vector<std::vector<int>>* innovIds, int* lastInnovId, std::vector<int> kinds, int maxIterationsFindNodeThresh, bool areRecurrentConnectionsAllowed) {	// return true = node created, false = nothing created
	// choose at random an enabled connection
	if ((int) connections.size() > 0) {	// if there is no connection, we cannot add a node!
		int iConn = rand() % (int) connections.size();
		int iterationNb = 0;
		while (iterationNb < maxIterationsFindNodeThresh && !connections[iConn].enabled) {
			iConn = rand() % (int) connections.size();
			iterationNb ++;
		}
		if (iterationNb < maxIterationsFindNodeThresh) {	// a connection has been found
			// disable former connection
			connections[iConn].enabled = false;
			
			// setup new node
			int newNodeId = (int) nodes.size();
			int inKind = rand() % (int) kind.size();
			int outKind = rand() % (int) kind.size();
			int funcId = findFuncId (inKind, outKind, kinds);
			nodes.push_back(Node(newNodeId, -1, inKind, outKind, funcId));	// no layer for the moment
			
			// build first connection
			int inNodeId = connections[iConn].inNodeId;
			int outNodeId = newNodeId;
			int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
			float weight = connections[iConn].weight;
			int inNodeRecu = connections[iConn].inNodeRecu;
			connections.push_back(Connection(innovId, inNodeId, outNodeId, inNodeRecu, weight, true));
			
			// build second connection
			inNodeId = newNodeId;
			outNodeId = connections[iConn].outNodeId;
			innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
			weight = Random_Float (- weightExtremumInit, weightExtremumInit);	// random number in [-weightExtremumInit; weightExtremumInit]
			int inNodeRecu = 0;	// the new node has been added without recurrency
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, inNodeRecu, true));

			// update layers
			if (connections[iConn].inNodeRecu > 0) {	// the connection was recurrent, so the layer has no effect
				nodes[newNodeId].layer = nodes[connections[iConn].inNodeId].layer;	// update newNodeId layer
			} else {									// else, the node is one layer further in the network
				nodes[newNodeId].layer = nodes[connections[iConn].inNodeId].layer + 1;	// update newNodeId layer
			}
			nodes[connections[iConn].outNodeId].layer = nodes[newNodeId].layer + 1;	// update outNodeId layer
			updateLayersRec(connections[iConn].outNodeId);	// recursively update layers

			// output nodes can have different nodes after updating layers: let's give output nodes the same output layer, the maximum one
			int maxLayer = nodes[nbBias + nbInput + nbOutput].layer;
			for (int i = nbBias + nbInput + nbOutput + 1; i < (int) nodes.size(); i++) {
				if (nodes[i].layer > maxLayer) {
					maxLayer = nodes[i].layer;
				}
			}
			for (int i = nbBias + nbInput; i < nbBias + nbInput + nbOutput; i++) {
				nodes[i].layer = maxLayer + 1;
			}

			return true;
		} else {
			std::cout << "Error : no active connection found while calling Genome::addNode" << std::endl;
			return false;	// no active connection found
		}
	} else {
		return false;	// there is no connection, cannot add a node
	}
}

void Genome::updateLayersRec(int nodeId) {
	for (int iConn = 0; iConn < (int) connections.size(); iConn++) {
		if (!(connections[iConn].inNodeRecu > 0) && connections[iConn].enabled && connections[iConn].inNodeId == nodeId) {
			int newNodeId = connections[iConn].outNodeId;
			nodes[newNodeId].layer = nodes[nodeId].layer + 1;
			updateLayersRec(newNodeId);
		}
	}
}

void Genome::drawNetwork(sf::Vector2u windowSize, float dotsRadius, std::string& font_path = "/usr/share/fonts/cantarell/Cantarell-VF.otf") {
	sf::RenderWindow window(sf::VideoMode(windowSize.x, windowSize.y), "VRNEAT - Titofra");
    
    sf::CircleShape dots[nodes.size()];
	sf::Text dotsText[nodes.size()];
	sf::Vertex lines[connections.size()][2];
	sf::Text mainText;
    
    // ### NODES ###
	sf::Font font;
	if (!font.loadFromFile(font_path)) {
		std::cout << "Error while loading font in 'Genome::drawNetwork'." << std::endl;
	}

	for (int i = 0; i < (int) nodes.size(); i++) {
		dots[i].setRadius(dotsRadius);
		dots[i].setFillColor(sf::Color::White);
		
		dotsText[i].setString(std::to_string(i));	// need of <iostream> for std::to_string function
		dotsText[i].setFillColor(sf::Color::White);
		dotsText[i].setCharacterSize(20);
		dotsText[i].setFont(font);
	}

	int nbLayer = nodes[nbBias + nbInput].layer + 1;

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
		if (connections[i].weight * connections[i].weight > maxWeight * maxWeight) {
			maxWeight = connections[i].weight;
		}
	}

	for (int i = 0; i < (int) connections.size(); i++) {
		sf::Color color;
		if (connections[i].enabled) {
			if (!(connections[i].inNodeRecu > 0)) {
				color = sf::Color::Green;
			} else {
				color = sf::Color::Blue;
			}
		} else {
			color = sf::Color::Red;
		}

		// weighted connections
		if (connections[i].weight / maxWeight > 0.0) {
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
