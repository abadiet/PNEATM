#include <VRNEAT/population.hpp>

using namespace vrneat;

Population::Population(int popSize, std::vector<int> nbInput, std::vector<int> nbOutput, std::vector<int> nbHiddenInit, float probConnInit, bool areRecurrentConnectionsAllowed, float weightExtremumInit, float speciationThreshInit, int threshGensSinceImproved): popSize(popSize), speciationThresh(speciationThreshInit), threshGensSinceImproved(threshGensSinceImproved), nbInput(nbInput), nbOutput(nbOutput), nbHiddenInit(nbHiddenInit), probConnInit(probConnInit), areRecurrentConnectionsAllowed(areRecurrentConnectionsAllowed), weightExtremumInit(weightExtremumInit) {
	generation = 0;
	N_connectionId = 0;
	fittergenome_id = -1;
	for (int i = 0; i < popSize; i++) {
		genomes.push_back(Genome(nbInput, nbOutput, nbHiddenInit, probConnInit, &innovIds, &lastInnovId, weightExtremumInit));
	}
}

void Population::loadInputs(void* inputs[]) {
	for (int i = 0; i < popSize; i++) {
		genomes[i].loadInputs(inputs);
	}
}

void Population::loadInputs(void* inputs[], int genome_id) {
	genomes[genome_id].loadInputs(inputs);
}

void Population::loadInput(void* input, int input_id) {
	for (int i = 0; i < popSize; i++) {
		genomes[i].loadInput(input, input_id);
	}
}

void Population::loadInput(void* input, int input_id, int genome_id) {
	genomes[genome_id].loadInput(input, input_id);
}

void Population::runNetwork() {
	for (int i = 0; i < popSize; i++) {
		genomes[i].runNetwork(activationFns);
	}
}

void Population::runNetwork(int genome_id) {
	genomes[genome_id].runNetwork(activationFns);
}

void Population::getOutputs(void* outputs[], int genome_id) {
	genomes[genome_id].getOutputs(outputs);
}

void* Population::getOutput(int output_id, int genome_id) {
	return genomes[genome_id].getOutput(input_id);
}

void Population::setFitness(float fitness, int genome_id) {
	genomes[genome_id].fitness = fitness;
}

void Population::speciate(int target, int targetThresh, float stepThresh, float a, float b, float c) {
	// reset species
	for (int i = 0; i < popSize; i++) {
		genomes[i].speciesId = -1;
	}
	
	// init species with leaders
	for (int iSpe = 0; iSpe < (int) species.size(); iSpe++) {
		if (!species[iSpe].isDead) {	// if the species is still alive
			int iMainGenome = species[iSpe].members[rand() % (int) species[iSpe].members.size()];	// select a random member to be the main genome of the species
			species[iSpe].members.clear();
			species[iSpe].members.push_back(iMainGenome);
			genomes[iMainGenome].speciesId = iSpe;
		}
	}
	
	// process the other genomes
	for (int genome_id = 0; genome_id < popSize; genome_id++) {
		if (genomes[genome_id].speciesId == -1) {	// if the genome not already belong to a species
			int speciesId = 0;
			while (speciesId < (int) species.size() && !(!species[speciesId].isDead && compareGenomes(species[speciesId].members[0], genome_id, a, b, c) < speciationThresh)) {
				speciesId ++;	// the genome cannot belong to this species, let's check the next one
			}
			if (speciesId == (int) species.size()) {	// no species found for the current genome, we have to create one new
				species.push_back(Species(speciesId));
			}
			species[speciesId].members.push_back(genome_id);
			genomes[genome_id].speciesId = speciesId;
		}
	}
	
	// check how many alive species we have
	int nbSpeciesAlive = 0;
	for (int iSpe = 0; iSpe < (int) species.size(); iSpe++) {
		if (!species[iSpe].isDead) {	// if the species is still alive
			nbSpeciesAlive ++;
		}
	}
	
	// update speciationThresh
	if (nbSpeciesAlive < target - targetThresh) {
		speciationThresh -= stepThresh;
	} else {
		if (nbSpeciesAlive > target + targetThresh) {
			speciationThresh += stepThresh;
		}
	}
	
	// update all the fitness
	updateFitnesses();
}

int Population::getConnectionId (int inNodeId, int outNodeId) {
	while ((int) connectionIds.size () < inNodeId) {
		connectionIds.push_back ({-1});
	}
	while ((int) connectionIds [inNodeId].size () < outNodeId) {
		connectionIds [inNodeId].push_back (-1);
	}
	if (connectionIds [inNodeId][outNodeId] == -1) {
		connectionIds [inNodeId][outNodeId] = N_connectionId;
		N_connectionId ++;
	}
	return connectionIds [inNodeId][outNodeId];
}

float Population::compareGenomes(int ig1, int ig2, float a, float b, float c) {
	int maxInnovId1 = 0;
	std::vector<int> connEnabled1;
	for (int i = 0; i < (int) genomes[ig1].connections.size(); i++) {
		if (genomes[ig1].connections[i].enabled) {
			connEnabled1.push_back(i);
			if (genomes[ig1].connections[i].innovId > maxInnovId1) {
				maxInnovId1 = genomes[ig1].connections[i].innovId;
			}
		}
	}
	
	int maxInnovId2 = 0;
	std::vector<int> connEnabled2;
	for (int i = 0; i < (int) genomes[ig2].connections.size(); i++) {
		if (genomes[ig2].connections[i].enabled) {
			connEnabled2.push_back(i);
			if (genomes[ig2].connections[i].innovId > maxInnovId2) {
				maxInnovId2 = genomes[ig2].connections[i].innovId;
			}
		}
	}
	
	int excessGenes = 0;
	int disjointGenes = 0;
	float sumDiffWeights = 0.0f;
	int nbCommonGenes = 0;
	
	for (int i1 = 0; i1 < (int) connEnabled1.size(); i1++) {
		if (genomes[ig1].connections[connEnabled1[i1]].innovId > maxInnovId2) {
			excessGenes += 1;
		} else {
			int i2 = 0;
			
			while (i2 < (int) connEnabled2.size() && genomes[ig2].connections[connEnabled2[i2]].innovId != genomes[ig1].connections[connEnabled1[i1]].innovId) {
				i2++;
			}
			if (i2 == (int) connEnabled2.size()) {
				disjointGenes += 1;
			} else {
				nbCommonGenes += 1;
				float diff = genomes[ig2].connections[connEnabled2[i2]].weight - genomes[ig1].connections[connEnabled1[i1]].weight;
				if (diff > 0) {
					sumDiffWeights += diff;
				} else {
					sumDiffWeights += -1 * diff;
				}
			}
		}
	}
	
	for (int i2 = 0; i2 < (int) connEnabled2.size(); i2++) {
		if (genomes[ig2].connections[connEnabled2[i2]].innovId > maxInnovId1) {
			excessGenes += 1;
		} else {
			int i1 = 0;
			while (i1 < (int) connEnabled1.size() && genomes[ig2].connections[connEnabled2[i2]].innovId != genomes[ig1].connections[connEnabled1[i1]].innovId) {
				i1++;
			}
			if (i1 == (int) connEnabled1.size()) {
				disjointGenes += 1;
			}
		}
	}
	
	if (nbCommonGenes > 0) {
		return (a * (float) excessGenes + b * (float) disjointGenes) / (float) std::max((int) connEnabled1.size(), (int) connEnabled2.size()) + c * sumDiffWeights / (float) nbCommonGenes;
	} else {
		return std::numeric_limits<float>::max();	// TODO: is there a better way?
	}
}

void Population::updateFitnesses() {
	fittergenome_id = 0;
	avgFitness = 0;
	avgFitnessAdjusted = 0;
	for (int i = 0; i < popSize; i++) {
		avgFitness += genomes[i].fitness;
		
		if (genomes[i].fitness > genomes[fittergenome_id].fitness) {
			fittergenome_id = i;
		}
	}
	
	avgFitness /= (float) popSize;
	
	for (int i = 0; i < (int) species.size(); i++) {
		if (!species[i].isDead) {
			species[i].sumFitness = 0;
			for (int j = 0; j < (int) species[i].members.size(); j++) {
				species[i].sumFitness += genomes[species[i].members[j]].fitness;
			}
			
			if (species[i].sumFitness / (float) species[i].members.size() > species[i].avgFitness) {	// the avgFitness of the species has increased
				species[i].gensSinceImproved  = 0;
			} else {
				species[i].gensSinceImproved ++;
			}
			
			species[i].avgFitness = species[i].sumFitness / (float) species[i].members.size();
			species[i].avgFitnessAdjusted = species[i].avgFitness / (float) species[i].members.size();
			
			avgFitnessAdjusted += species[i].avgFitness;
		}
	}
	
	avgFitnessAdjusted /= (float) popSize;
	
	for (int i = 0; i < (int) species.size(); i++) {
		if (!species[i].isDead) {
			if (species[i].gensSinceImproved < threshGensSinceImproved) {
				species[i].allowedOffspring = (int) ((float) species[i].members.size() * species[i].avgFitnessAdjusted / (avgFitnessAdjusted + std::numeric_limits<float>::min()));	// note that (int) 0.9f == 0.0f	// numeric_limits<float>::min() = minimum positive value of float
			} else {
				species[i].allowedOffspring = 0;
			}
		}
	}
}

void Population::crossover(bool elitism) {
	std::vector<Genome> newGenomes;
	
	if (elitism) {	// elitism mode on = we conserve during generations the fitter genome
		Genome newGenome(nbInput, nbOutput, nbHiddenInit, probConnInit, &innovIds, &lastInnovId, weightExtremumInit);
		newGenome.nodes = genomes[fittergenome_id].nodes;
		newGenome.connections = genomes[fittergenome_id].connections;
		newGenome.speciesId = genomes[fittergenome_id].speciesId;
		newGenomes.push_back(newGenome);
	}
	
	for (int iSpe = 0; iSpe < (int) species.size() ; iSpe++) {
		for (int k = 0; k < species[iSpe].allowedOffspring; k++) {
			// choose pseudo-randomly two parents. Don't care if they're identical as the child will be mutated...
			int iParent1 = selectParent(iSpe);
			int iParent2 = selectParent(iSpe);
			
			// clone the fitter
			Genome newGenome(nbInput, nbOutput, nbHiddenInit, probConnInit, &innovIds, &lastInnovId, weightExtremumInit);
			int iMainParent;
			int iSecondParent;
			if (genomes[iParent1].fitness > genomes[iParent2].fitness) {
				iMainParent = iParent1;
				iSecondParent = iParent2;
			} else {
				iMainParent = iParent2;
				iSecondParent = iParent1;
			}
			newGenome.nodes = genomes[iMainParent].nodes;
			
			newGenome.connections = genomes[iMainParent].connections;
			newGenome.speciesId = iSpe;
			
			// connections shared by both of the parents must be randomly wheighted
			for (int iMainParentConn = 0; iMainParentConn < (int) genomes[iMainParent].connections.size(); iMainParentConn++) {
				for (int iSecondParentConn = 0; iSecondParentConn < (int) genomes[iSecondParent].connections.size(); iSecondParentConn++) {
					if (genomes[iMainParent].connections[iMainParentConn].innovId == genomes[iSecondParent].connections[iSecondParentConn].innovId) {
						if (rand() % 2 == 0) {	// 50 % of chance for each parent, newGenome already have the wheight of MainParent
							newGenome.connections[iMainParentConn].weight = genomes[iSecondParent].connections[iSecondParentConn].weight;
						}
					}
				}
			}
			
			newGenomes.push_back(newGenome);
		}
	}
	
	int previousSize = (int) newGenomes.size();
	// add genomes if some are missing
	for (int k = 0; k < popSize - previousSize; k++) {
		newGenomes.push_back(Genome(nbInput, nbOutput, nbHiddenInit, probConnInit, &innovIds, &lastInnovId, weightExtremumInit));
	}
	
	// or remove some genomes if there is too many genomes
	for (int k = 0; k < previousSize - popSize; k++) {
		newGenomes.pop_back();
	}
	
	genomes = newGenomes;
	
	// reset species members
	for (int i = 0; i < (int) species.size(); i++) {
		species[i].members.clear();
		species[i].isDead = true;
	}
	for (int i = 0; i < popSize; i++) {
		if (genomes[i].speciesId > -1) {
			species[genomes[i].speciesId].members.push_back(i);
			species[genomes[i].speciesId].isDead = false;	// empty species will stay to isDead = true
		}
	}
	
	fittergenome_id = -1;	// avoid to missuse fittergenome_id
	
	generation ++;
}

int Population::selectParent(int iSpe) {
	/* Chooses player from the population to return randomly(considering fitness). This works by randomly choosing a value between 0 and the sum of all the fitnesses then go through all the dots and add their fitness to a running sum and if that sum is greater than the random value generated that dot is chosen since players with a higher fitness function add more to the running sum then they have a higher chance of being chosen */
	// build a random threshold in [0, sumFitness)
	float randThresh = (float) rand()/(float) (RAND_MAX);
	while (randThresh < 1.0f + 1e-10 && randThresh > 1.0f - 1e-10) {	// == 1
		randThresh = (float) rand()/(float) (RAND_MAX);
	}
	randThresh *= species[iSpe].sumFitness;
	
	float runningSum = 0.0f;
	
	for (int i = 0; i < (int) species[iSpe].members.size(); i++) {
		runningSum += genomes[species[iSpe].members[i]].fitness;
		if (runningSum > randThresh) {
			return species[iSpe].members[i];
		}
	}
	std::cout << "Error : don't find a parent during crossover." << std::endl;
	return -1;	// impossible
}

void Population::mutate(float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor, float addConnectionThresh, int maxIterationsFindConnectionThresh, float reactivateConnectionThresh, float addNodeThresh, int maxIterationsFindNodeThresh) {
	for (int i = 0; i < popSize; i++) {
		genomes[i].mutate(&innovIds, &lastInnovId, areRecurrentConnectionsAllowed, mutateWeightThresh, mutateWeightFullChangeThresh, mutateWeightFactor, addConnectionThresh, maxIterationsFindConnectionThresh, reactivateConnectionThresh, addNodeThresh, maxIterationsFindNodeThresh);
	}
}

void Population::drawNetwork(int genome_id, sf::Vector2u windowSize, float dotsRadius) {
	genomes[genome_id].drawNetwork(windowSize, dotsRadius);
}

void Population::printInfo(bool extendedGlobal, bool printSpecies, bool printGenomes, bool extendedGenomes) {
	std::cout << "GENERATION " << generation << std::endl;
	
	std::cout << "	" << "Global" << std::endl;
	std::cout << "	" << "	" << "Average fitness: " << avgFitness << std::endl;
	std::cout << "	" << "	" << "Average fitness (adjusted): " << avgFitnessAdjusted << std::endl;
	std::cout << "	" << "	" << "Best fitness: " << genomes[fittergenome_id].fitness << std::endl;
	if (extendedGlobal) {
		std::cout << "	" << "	" << "Population size: " << popSize << std::endl;
		std::cout << "	" << "	" << "Number of inputs: " << nbInput << std::endl;
		std::cout << "	" << "	" << "Number of outputs: " << nbOutput << std::endl;
		std::cout << "	" << "	" << "Speciation threshold: " << speciationThresh << std::endl;
		std::cout << "	" << "	" << "Are recurrent connections allowed: ";
		if (areRecurrentConnectionsAllowed) {
			std::cout << "yes" << std::endl;
		} else {
			std::cout << "no" << std::endl;
		}
		std::cout << "	" << "	" << "When initializing a new genome" << std::endl;
		std::cout << "	" << "	" << "	" << "Number of hidden nodes: " << nbHiddenInit << std::endl;
		std::cout << "	" << "	" << "	" << "Proability of a connection to be created: " << probConnInit << std::endl;
		std::cout << "	" << "	" << "	" << "Weight bounds: " << weightExtremumInit << std::endl;
	
	}
	
	if (printSpecies) {
		std::cout << "	" << "Species [id, average fitness, average fitness (adjusted), number of allowed offspring(s), number of generations since improved, number of members, dead]" << std::endl;
		for (int i = 0; i < (int) species.size(); i++) {
			std::cout << "	" << "	" << species[i].id << "	" << species[i].avgFitness << "	" << species[i].avgFitnessAdjusted << "	" << species[i].allowedOffspring << "	" << species[i].gensSinceImproved << "	" << (int) species[i].members.size() << "	";
			if (species[i].isDead) {
				std::cout << "yes" << std::endl;
			} else {
				std::cout << "no" << std::endl;
			}
		}
	}
	
	if (printGenomes) {
		std::cout << "	" << "Genomes [id, fitness, id of the species]" << std::endl;
		for (int i = 0; i < popSize; i++) {
			std::cout << "	" << "	" << i << "	" << genomes[i].fitness << "	" << genomes[i].speciesId << std::endl;
			if (extendedGenomes) {
				for (int k = 0; k < (int) genomes[i].connections.size(); k++) {
					std::cout << "	" << "	" << "	" << genomes[i].connections[k].inNodeId << " -> " << genomes[i].connections[k].outNodeId << "	(W: " << genomes[i].connections[k].weight << ", Innov: " << genomes[i].connections[k].innovId << ")";
					if (genomes[i].connections[k].isRecurrent) {
						std::cout << " R ";
					}
					if (!genomes[i].connections[k].enabled) {
						std::cout << " D ";
					}
					std::cout << std::endl;
				}
			}
		}
	}
}

void Population::save(const std::string filepath){
	std::ofstream fileobj(filepath);

	if (fileobj.is_open()){
		for (int k = 0; k < (int) innovIds.size(); k++){
			for (int j = 0; j < (int) innovIds[k].size(); j++){
				fileobj << innovIds[k][j] << ",";
			}
			fileobj << ";";
		}
		fileobj << "\n";

		fileobj << lastInnovId << "\n";
		fileobj << popSize << "\n";
		fileobj << speciationThresh << "\n";
		fileobj << threshGensSinceImproved << "\n";
		fileobj << nbInput << "\n";
		fileobj << nbOutput << "\n";
		fileobj << nbHiddenInit << "\n";
		fileobj << probConnInit << "\n";
		fileobj << areRecurrentConnectionsAllowed << "\n";
		fileobj << weightExtremumInit << "\n";
		fileobj << generation << "\n";
		fileobj << avgFitness << "\n";
		fileobj << avgFitnessAdjusted << "\n";
		fileobj << fittergenome_id << "\n";

		for (int k = 0; k < (int) genomes.size(); k++){
			fileobj << genomes[k].fitness << "\n";
			fileobj << genomes[k].speciesId << "\n";


			for (int j = 0; j < (int) genomes[k].nodes.size(); j++){
				fileobj << genomes[k].nodes[j].id << ",";
				fileobj << genomes[k].nodes[j].layer << ",";
				fileobj << genomes[k].nodes[j].sumInput << ",";
				fileobj << genomes[k].nodes[j].sumOutput << ",";
			}
			fileobj << "\n";

			for (int j = 0; j < (int) genomes[k].connections.size(); j++){
				fileobj << genomes[k].connections[j].innovId << ",";
				fileobj << genomes[k].connections[j].inNodeId << ",";
				fileobj << genomes[k].connections[j].outNodeId << ",";
				fileobj << genomes[k].connections[j].weight << ",";
				fileobj << genomes[k].connections[j].enabled << ",";
				fileobj << genomes[k].connections[j].isRecurrent << ",";
			}
			fileobj << "\n";
		}

		fileobj.close();
	}
}

void Population::load(const std::string filepath){
	std::ifstream fileobj(filepath);

	if (fileobj.is_open()){

		std::string line;
		size_t pos = 0;

		if (getline(fileobj, line)){
			innovIds.clear();
			size_t pos_sep = line.find(';');
			while (pos_sep != std::string::npos) {
				innovIds.push_back({});
				std::string sub_line = line.substr(0, pos_sep - 1);
				pos = sub_line.find(',');
				while (pos != std::string::npos) {
					innovIds.back().push_back(stoi(sub_line.substr(0, pos)));
					sub_line = sub_line.substr(pos + 1);
					pos = sub_line.find(',');
				}
				line = line.substr(pos_sep + 1);
				pos_sep = line.find(';');
			}
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}

		if (getline(fileobj, line)){
			lastInnovId = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			popSize = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			speciationThresh = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			threshGensSinceImproved = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			nbInput = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			nbOutput = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			nbHiddenInit = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			probConnInit = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			areRecurrentConnectionsAllowed = (line == "1");
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			weightExtremumInit = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			generation = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			avgFitness = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			avgFitnessAdjusted = stof(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		if (getline(fileobj, line)){
			fittergenome_id = stoi(line);
		} else {
			std::cout << "Error while loading model" << std::endl;
			throw 0;
		}
		genomes.clear();
		while (getline(fileobj, line)){
			genomes.push_back(Genome(nbInput, nbOutput, nbHiddenInit, probConnInit, &innovIds, &lastInnovId, weightExtremumInit));

			genomes.back().fitness = stof(line);
			if (getline(fileobj, line)){
				genomes.back().speciesId = stoi(line);
			} else {
				std::cout << "Error while loading model" << std::endl;
				throw 0;
			}
			if (getline(fileobj, line)){
				genomes.back().nodes.clear();
				pos = line.find(',');
				while (pos != std::string::npos) {
					genomes.back().nodes.push_back(Node());

					genomes.back().nodes.back().id = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().nodes.back().layer = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().nodes.back().sumInput = (float) stod(line.substr(0, pos));	// stdof's range is not the float's one... weird, anyway it works with stod of course

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().nodes.back().sumOutput = (float) stod(line.substr(0, pos));	// stdof's range is not the float's one... weird, anyway it works with stod of course

					line = line.substr(pos + 1);
					pos = line.find(',');
				}
			} else {
				std::cout << "Error while loading model" << std::endl;
				throw 0;
			}
			if (getline(fileobj, line)){
				genomes.back().connections.clear();
				pos = line.find(',');
				while (pos != std::string::npos) {
					genomes.back().connections.push_back(Connection());

					genomes.back().connections.back().innovId = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().inNodeId = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().outNodeId = stoi(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().weight = stof(line.substr(0, pos));

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().enabled = (line.substr(0, pos) == "1");

					line = line.substr(pos + 1);
					pos = line.find(',');
					if (pos == std::string::npos) {
						std::cout << "Error while loading model" << std::endl;
						throw 0;
					}
					genomes.back().connections.back().isRecurrent = (line.substr(0, pos) == "1");

					line = line.substr(pos + 1);
					pos = line.find(',');
				}
			} else {
				std::cout << "Error while loading model" << std::endl;
				throw 0;
			}
		}

		fileobj.close();
	} else {
		std::cout << "Error while loading model" << std::endl;
		throw 0;
	}
}
