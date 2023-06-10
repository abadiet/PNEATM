#include <PNEATM/genome.hpp>

using namespace pneatm;

template <typename... Args>
Genome<Args...>::Genome (std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_init, std::vector<void*> resetValues, std::vector<std::vector<std::vector<std::function <void* (void*)>>>> activationFns, innovation_t* conn_innov, unsigned int N_ConnInit, float probRecuInit, float weightExtremumInit, unsigned int maxRecuInit) :
	weightExtremumInit (weightExtremumInit),
	activationFns (activationFns),
	resetValues (resetValues)

{
	N_types = (unsigned int) activationFns.size ();
	speciesId = -1;

	// NODES
	// bias
	nbBias = 0;
	for (size_t i = 0; i < bias_sch.size (); i++) {
		for (size_t k = 0; k < bias_sch [i]; k++) {
			// get Node<T_in, T_out>
			nodes.push_back(CreateNode<Args...>::get (i, i));

			// input Nodes' activation's functions are the identity
			std::function<void* (void*)> generic_func = [] (void* input) {return input;};
			
			// setup the node
			nodes.back ()->id = nbBias;
			nodes.back ()->layer = 0;
			nodes.back ()->index_T_in = i;
			nodes.back ()->index_T_out = i;
			nodes.back ()->setActivationFn (generic_func);
			nodes.back ()->setResetValue (bias_init [i]);	// for bias nodes, init and reset value are the same
			nodes.back ()->reset ();	// set the init value

			nbBias ++;
		}
	}
	// input
	nbInput = 0;
	for (size_t i = 0; i < inputs_sch.size (); i++) {
		for (size_t k = 0; k < inputs_sch [nbBias + nbInput]; k++) {
			// get Node<T_in, T_out>
			nodes.push_back(CreateNode<Args...>::get (i, i));

			// input Nodes' activation's functions are the identity
			std::function<void* (void*)> generic_func = [] (void* input) {return input;};
			
			// setup the node
			nodes.back ()->id = nbBias + nbInput;
			nodes.back ()->layer = 0;
			nodes.back ()->index_T_in = i;
			nodes.back ()->index_T_out = i;
			nodes.back ()->setActivationFn (generic_func);
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
		for (size_t k = 0; k < outputs_sch [nbOutput]; k++) {
			// get Node<T_in, T_out>
			nodes.push_back(CreateNode<Args...>::get (i, i));

			// output Nodes' activation's functions are the identity
			std::function<void* (void*)> generic_func = [] (void* input) {return input;};
			
			// setup the node
			nodes.back ()->id = nbBias + nbInput + nbOutput;
			nodes.back ()->layer = outputLayer;
			nodes.back ()->index_T_in = i;
			nodes.back ()->index_T_out = i;
			nodes.back ()->setActivationFn (generic_func);
			nodes.back ()->setResetValue (resetValues [i]);

			nbOutput ++;
		}
	}
	// hidden
	unsigned int nbHidden = 0;
	for (size_t i = 0; i < hiddens_sch_init.size (); i++) {
		for (size_t j = 0; j < hiddens_sch_init [i].size (); j++) {
			for (size_t k = 0; k < hiddens_sch_init [i][nbHidden]; k++) {
				// get Node<T_in, T_out>
				nodes.push_back(CreateNode<Args...>::get (i, j));

				// activation function for the hidden node
				std::function<void* (void*)> generic_func = activationFns [i][j][
					rand () % activationFns [i][j].size ()
				];

				// setup the node
				nodes.back ()->id = nbBias + nbInput + nbOutput + nbHidden;
				nodes.back ()->layer = 1;
				nodes.back ()->index_T_in = i;
				nodes.back ()->index_T_out = j;
				nodes.back ()->setActivationFn (generic_func);
				nodes.back ()->setResetValue (resetValues [i]);

				nbHidden ++;
			}
		}
	}
	
	// CONNECTIONS
	unsigned int iConn = 0;
	while (iConn < N_ConnInit) {
		// inNodeId and outNodeId
		unsigned int inNodeId = rand () % nodes.size ();
		unsigned int outNodeId = rand () % nodes.size ();

		// inNodeRecu
		unsigned int inNodeRecu = 0;
		if (Random_Float (0.0f, 1.0f, true, false) < probRecuInit) {
			inNodeRecu = rand () % (maxRecuInit + 1);
		}

		if (CheckNewConnectionValidity (inNodeId, outNodeId, inNodeRecu)) {
			// innovId
			const unsigned int innov_id = conn_innov->getInnovId (inNodeId, outNodeId, inNodeRecu);

			// weight
			const float weight = Random_Float (- weightExtremumInit, weightExtremumInit);	// random number in [-weightExtremumInit; weightExtremumInit]

			connections.push_back(Connection (innov_id, inNodeId, outNodeId, inNodeRecu, weight, true));

			if (inNodeRecu == 0) {
				UpdateLayers (inNodeId);
			}

			iConn ++;
		}
	}
}

template <typename... Args>
template <typename T_in>
void Genome<Args...>::loadInputs (T_in inputs []) {
	for (unsigned int i = 0; i < nbInput; i++) {
		nodes[i + nbBias]->setInput (inputs [i]);
	}
}

template <typename... Args>
template <typename T_in>
void Genome<Args...>::loadInput (T_in input, int input_id) {
	nodes[input_id + nbBias]->setInput (input);
}

template <typename... Args>
void Genome<Args...>::runNetwork() {
	/* Process all input and output. For that, it "scans" each layer from the inputs to the last hidden's layer to calculate input with already known value. */ 

	// reset input
	for (size_t i = nbBias + nbInput; i < nodes.size(); i++) {
		nodes [i]->reset ();
	}
	
	int lastLayer = nodes[nbBias + nbInput]->layer;
	
	for (int ilayer = 0; ilayer < lastLayer; ilayer++) {
		// process nodes[*]->input
		for (size_t i = 0; i < connections.size (); i++) {
			if (connections [i].enabled && nodes [connections [i].outNodeId]->layer == ilayer) {	// if the connections still exist and is pointing on the current layer
				if (connections [i].inNodeRecu == 0) {
					nodes [connections [i].outNodeId]->AddToInput (
						nodes [connections [i].inNodeId]->getOutput (),
						connections [i].weight
					);
				} else {	// is recurent
					if (connections[i].inNodeRecu < (unsigned int) prevNodes.size () + 1) {
						nodes[connections[i].outNodeId]->AddToInput (
							prevNodes [(unsigned int) prevNodes.size () - connections [i].inNodeRecu][connections [i].inNodeId]->getOutput (),
							connections [i].weight
						);
					} else {
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
	prevNodes.push_back (nodes);
}

template <typename... Args>
template <typename T_out>
void Genome<Args...>::getOutputs (T_out outputs []) {
	for (int i = 0; i < nbOutput; i++) {
		outputs [i] = (T_out) nodes [nbBias + nbInput + i]->getOutput ();
	}
}

template <typename... Args>
template <typename T_out>
T_out Genome<Args...>::getOutput (int output_id) {
	return (T_out) nodes[nbBias + nbInput + output_id]->getOutput ();
}

template <typename... Args>
void Genome<Args...>::mutate(innovation_t* conn_innov, unsigned int maxRecurrency, float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor, float addConnectionThresh, int maxIterationsFindConnectionThresh, float reactivateConnectionThresh, float addNodeThresh, int maxIterationsFindNodeThresh, float addTranstypeThresh) {
	// WEIGHTS
	if (Random_Float (0.0f, 1.0f, true, false) < mutateWeightThresh) {
		MutateWeights (mutateWeightFullChangeThresh, mutateWeightFactor);
	}
	
	// NODES
	if (Random_Float (0.0f, 1.0f, true, false) < addNodeThresh) {
		AddNode (conn_innov, maxIterationsFindNodeThresh);
	}

	// TRANSTYPE (aka add a bi-typed node and two connections)
	if (Random_Float (0.0f, 1.0f, true, false) < addTranstypeThresh) {
		AddTranstype (conn_innov, maxRecurrency, maxIterationsFindNodeThresh);
	}

	// CONNECTIONS
	if (Random_Float (0.0f, 1.0f, true, false) < addConnectionThresh) {
		AddConnection (conn_innov, maxRecurrency, maxIterationsFindConnectionThresh, reactivateConnectionThresh);
	}

	// TODO? We might updateLayers here, but is this useful?
}

template <typename... Args>
bool Genome<Args...>::CheckNewConnectionValidity (unsigned int inNodeId, unsigned int outNodeId, unsigned int inNodeRecu, unsigned int* disabled_conn_id) {
	if (nodes [inNodeId]->index_T_out == nodes [outNodeId]->index_T_in) return false;	// connections should link two same objects

	for (size_t i = 0; i < connections.size (); i++) {
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
		if (connections [iConn].inNodeId = outNodeId && connections [iConn].inNodeRecu == 0) {
			if (CheckNewConnectionCircle (inNodeId, connections [iConn].outNodeId)) {
				return true;
			}
		}
	}
	return false;

}

template <typename... Args>
void Genome<Args...>::MutateWeights (float mutateWeightFullChangeThresh, float mutateWeightFactor) {
	for (size_t i = 0; i < connections.size(); i++) {
		if (Random_Float (0.0f, 1.0f, true, false) < mutateWeightFullChangeThresh) {
			// reset weight
			connections [i].weight = Random_Float (- weightExtremumInit, weightExtremumInit);
		} else {
			// pertub weight
			connections [i].weight *= Random_Float (- mutateWeightFactor, mutateWeightFactor);
		}
	}
}

template <typename... Args>
bool Genome<Args...>::AddConnection (innovation_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindConnectionThresh, float reactivateConnectionThresh) {	// return true if the process ended well, false in the other case
	// find valid node pair
	unsigned int iterationNb = 1;
	unsigned int inNodeId = rand() % nodes.size();
	unsigned int outNodeId = rand() % nodes.size();
	unsigned int inNodeRecu = rand() % (maxRecurrency + 1);
	int disabled_conn_id = -1;
	while (
		iterationNb < maxIterationsFindConnectionThresh
		&& CheckNewConnectionValidity (inNodeId, outNodeId, inNodeId, &disabled_conn_id)
	) {
		inNodeId = rand() % nodes.size();
		outNodeId = rand() % nodes.size();
		inNodeRecu = rand() % (maxRecurrency + 1);
		iterationNb ++;
	}
	
	if (iterationNb < maxIterationsFindConnectionThresh) {	// a valid connection has been found
		// mutating
		if (disabled_conn_id >= 0) {	// it is a former connection
			if (Random_Float (0.0f, 1.0f, true, false) < reactivateConnectionThresh) {
				connections [disabled_conn_id].enabled = true;	// former connection is reactivated
				return true;
			} else {
				return true;	// return true even no connection has been change because process ended well
			}
		} else {
			// innovId
			const unsigned int innov_id = conn_innov->getInnovId (inNodeId, outNodeId, inNodeRecu);

			// weight
			const float weight = Random_Float (- weightExtremumInit, weightExtremumInit);	// random number in [-weightExtremumInit; weightExtremumInit]

			connections.push_back(Connection (innov_id, inNodeId, outNodeId, inNodeRecu, weight, true));
			
			return true;
		}
	} else {
		return false;	// cannot find a valid connection
	}
}

template <typename... Args>
bool Genome<Args...>::AddNode (innovation_t* conn_innov, unsigned int maxIterationsFindNodeThresh) {	// return true = node created, false = nothing created
	// choose at random an enabled connection
	if (connections.size() > 0) {	// if there is no connection, we cannot add a node!
		unsigned int iConn = rand() % connections.size();
		unsigned int iterationNb = 0;
		while (iterationNb < maxIterationsFindNodeThresh && !connections [iConn].enabled) {
			iConn = rand() % connections.size();
			iterationNb ++;
		}
		if (iterationNb < maxIterationsFindNodeThresh) {	// a connection has been found
			// disable former connection
			connections [iConn].enabled = false;
			
			// setup new node
			const unsigned int newNodeId = (unsigned int) nodes.size ();
			const unsigned int iT_in = nodes [connections [iConn].inNodeId]->index_T_in;
			const unsigned int iT_out = nodes [connections [iConn].outNodeId]->index_T_out;

			// get Node<T_in, T_out>
			nodes.push_back(CreateNode<Args...>::get (iT_in, iT_out));

			// activation function
			std::function<void* (void*)> generic_func = activationFns [iT_in][iT_out][
				rand () % activationFns [iT_in][iT_out].size ()
			];

			// setup the node
			nodes.back ()->id = newNodeId;
			nodes.back ()->layer = -1;	// no layer for now
			nodes.back ()->index_T_in = iT_in;
			nodes.back ()->index_T_out = iT_out;
			nodes.back ()->setActivationFn (generic_func);
			nodes.back ()->setResetValue (resetValues [iT_in]);

			// build first connection
			int inNodeId = connections [iConn].inNodeId;
			int outNodeId = newNodeId;
			unsigned int inNodeRecu = connections [iConn].inNodeRecu;
			unsigned int innovId = conn_innov->getInnovId (inNodeId, outNodeId, inNodeRecu);
			float weight = connections [iConn].weight;

			connections.push_back (Connection (innovId, inNodeId, outNodeId, inNodeRecu, weight, true));
			
			// build second connection
			inNodeId = newNodeId;
			outNodeId = connections [iConn].outNodeId;
			inNodeRecu = 0;
			innovId = conn_innov->getInnovId (inNodeId, outNodeId, inNodeRecu);
			weight = Random_Float (- weightExtremumInit, weightExtremumInit);	// random number in [-weightExtremumInit; weightExtremumInit]

			connections.push_back (Connection (innovId, inNodeId, outNodeId, inNodeRecu, weight, true));

			// update layers
			if (connections [iConn].inNodeRecu > 0) {	// the connection was recurrent, so the layer has no effect
				nodes [newNodeId]->layer = nodes [connections [iConn].outNodeId]->layer - 1;	// update newNodeId layer
			} else {									// else, the node is one layer further in the network
				nodes [newNodeId]->layer = nodes [connections[iConn].inNodeId]->layer + 1;	// update newNodeId layer
				nodes [connections [iConn].outNodeId]->layer = nodes[newNodeId]->layer + 1;	// update outNodeId layer
				UpdateLayers (connections [iConn].outNodeId);	// update other layers
			}

			return true;
		} else {
			return false;	// no active connection found
		}
	} else {
		return false;	// there is no connection, cannot add a node
	}
}

template <typename... Args>
bool Genome<Args...>::AddTranstype (innovation_t* conn_innov, unsigned int maxRecurrency, unsigned int maxIterationsFindNodeThresh) {
	if (N_types > 1) {
		// Add bi-typed node
		const unsigned int newNodeId = (unsigned int) nodes.size ();
		const unsigned int iT_in = rand () % N_types;
		unsigned int iT_out = rand () % N_types;
		while (iT_out == iT_in) {
			iT_out = rand () % N_types;
		}

		// get Node<T_in, T_out>
		nodes.push_back(CreateNode<Args...>::get (iT_in, iT_out));

		// activation function
		std::function<void* (void*)> generic_func = activationFns [iT_in][iT_out][
			rand () % activationFns [iT_in][iT_out].size ()
		];

		// setup the node
		nodes.back ()->id = newNodeId;
		nodes.back ()->layer = -1;	// no layer for now
		nodes.back ()->index_T_in = iT_in;
		nodes.back ()->index_T_out = iT_out;
		nodes.back ()->setActivationFn (generic_func);
		nodes.back ()->setResetValue (resetValues [iT_in]);

		// Add the first connection
		unsigned int inNodeId = rand() % nodes.size();
		unsigned int inNodeRecu = rand () % (maxRecurrency + 1);
		unsigned int iterationNb = 0;
		while (
			iterationNb < maxIterationsFindNodeThresh
			&& (
				nodes [inNodeId]->index_T_out != iT_in
				|| (
					inNodeId >= nbBias + nbInput && inNodeId < nbBias + nbInput + nbOutput	// cannot build a non recurrent connection with an output node as the input's connection
					&& inNodeRecu > 0
				)
			)
		) {
			inNodeId = rand() % nodes.size();
			iterationNb ++;
		}
		if (iterationNb == maxIterationsFindNodeThresh) return false;

		unsigned int innov_id = conn_innov->getInnovId (inNodeId, newNodeId, inNodeRecu);
		float weight = Random_Float (- weightExtremumInit, weightExtremumInit);	// random number in [-weightExtremumInit; weightExtremumInit]

		connections.push_back(Connection (innov_id, inNodeId, newNodeId, inNodeRecu, weight, true));

		// Add the second connection
		unsigned int outNodeId = rand() % nodes.size();
		inNodeRecu = 0;
		iterationNb = 0;
		while (
			iterationNb < maxIterationsFindNodeThresh
			&& (
				nodes [outNodeId]->index_T_in != iT_out
				|| CheckNewConnectionCircle (newNodeId, outNodeId)
			)
		) {
			outNodeId = rand() % nodes.size();
			iterationNb ++;
		}
		if (iterationNb == maxIterationsFindNodeThresh) return false;

		innov_id = conn_innov->getInnovId (newNodeId, outNodeId, inNodeRecu);
		weight = Random_Float (- weightExtremumInit, weightExtremumInit);	// random number in [-weightExtremumInit; weightExtremumInit]

		connections.push_back(Connection (innov_id, newNodeId, outNodeId, inNodeRecu, weight, true));

		return true;
	} else {
		return false;	// there is only one type of object
	}
}

template <typename... Args>
void Genome<Args...>::UpdateLayers_Recursive (int inNodeId) {
	for (size_t iConn = 0; iConn < connections.size (); iConn ++) {
		if (
			!(connections [iConn].inNodeRecu > 0)
			&& connections [iConn].enabled
			&& connections [iConn].inNodeId == inNodeId
		) {
			unsigned int newNodeId = connections [iConn].outNodeId;
			nodes [newNodeId]->layer = nodes [inNodeId]->layer + 1;

			UpdateLayers_Recursive (newNodeId);
		}
	}
}

template <typename... Args>
void Genome<Args...>::UpdateLayers (int inNodeId) {
	// Update layers
	UpdateLayers_Recursive (inNodeId);

	// this might move some output's node, let's homogenize that
	unsigned int outputLayer = nodes [nbBias + nbInput]->layer;
	for (size_t i = nbBias + nbInput + 1; i < nbBias + nbInput + nbOutput; i ++) {
		// check among the outputs which one is the highest and set the output layer to it
		if (nodes [i]->layer > outputLayer) {
			outputLayer = nodes [i]->layer;
		}
	}
	for (size_t i = nbBias + nbInput + nbOutput; i < nodes.size (); i ++) {
		// check among the hiddens which one is the highest and set the output layer to it + 1
		if (nodes [i]->layer > outputLayer) {
			outputLayer = nodes [i]->layer + 1;
		}
	}
	for (size_t i = nbBias + nbInput + 1; i < nbBias + nbInput + nbOutput; i ++) {
		// new layer!
		nodes [i]->layer = outputLayer;
	}
}

/*
template <typename... Args>
void Genome<Args...>::drawNetwork(sf::Vector2u windowSize, float dotsRadius, std::string& font_path = "/usr/share/fonts/cantarell/Cantarell-VF.otf") {
	sf::RenderWindow window(sf::VideoMode(windowSize.x, windowSize.y), "PNEATM - Titofra");
    
    sf::CircleShape dots[nodes.size()];
	sf::Text dotsText[nodes.size()];
	sf::Vertex lines[connections.size()][2];
	sf::Text mainText;
    
    // ### NODES ###
	sf::Font font;
	if (!font.loadFromFile(font_path)) {
		std::cout << "Error while loading font in 'Genome<Args...>::drawNetwork'." << std::endl;
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
*/