#ifndef SPECIES_HPP
#define SPECIES_HPP

#include <PNEATM/Connection/connection.hpp>
#include <PNEATM/genome.hpp>
#include <PNEATM/utils.hpp>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <cstring>
#include <fstream>

/* HEADER */

namespace pneatm {

enum distanceFn {
	CONVENTIONAL,
	EUCLIDIAN
};

/**
 * @brief A template class representing a species.
 * @tparam Args Variadic template arguments that contains all the manipulated types.
 */
template <typename... Args>
class Species {
	public:
		/**
		 * @brief Constructor for the Species class.
		 * @param id the species ID.
		 * @param connections A unordered map of connection that define the species traits. Will be used to process the distance between the species and genomes.
		 * @param dstType The distance algorithm to use:\n	- `CONVENTIONAL`: algorithm used in the original NEAT\n	- `EUCLIDIAN`: euclidian distance in the connections's space
		 */
		Species (unsigned int id, const std::unordered_map<unsigned int, Connection>& connections, distanceFn dstType);

		/**
		 * @brief Constructor for the Species class from an input file stream.
		 * @param inFile The input file stream.
		 */
		Species (std::ifstream& inFile);

		/**
		 * @brief Destructor for the Species class.
		 */
		~Species () {};

		/**
		 * @brief Return the distance between a given genome and the species.
		 * @param genome A reference to the genome for which to calculate the distance.
		 * @param a Coefficient for computing the excess genes contribution to the distance [for CONVENTIONAL distance only]. (default is 1.0)
		 * @param b Coefficient for computing the disjoint genes contribution to the distance [for CONVENTIONAL distance only]. (default is 1.0)
		 * @param c Coefficient for computing the average weight difference contribution to the distance [for CONVENTIONAL distance only]. (default is 0.4)
		 * @return The distance between a given genome and the species
		 */
		double distanceWith (const std::unique_ptr<Genome<Args...>>& genome, double a = 1.0, double b = 1.0, double c = 0.4);

		/**
		 * @brief Print information on the species.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (const std::string& prefix = "") const;

		/**
		 * @brief Serialize the Species instance to an output file stream.
		 * @param outFile The output file stream to which the Species instance will be written.
		 */
		void serialize (std::ofstream& outFile) const;

		/**
		 * @brief Deserialize a Species instance from an input file stream.
		 * @param inFile The input file stream from which the Species instance will be read.
		 */
		void deserialize (std::ifstream& inFile);

	private:
		unsigned int id;
		distanceFn dstType;
		std::unordered_map<unsigned int, Connection> connections;
		double avgFitness;
		double avgFitnessAdjusted;
		int allowedOffspring;
		double sumFitness;
		unsigned int gensSinceImproved;
		bool isDead;
		std::vector<unsigned int> members;

		// distance functions
		double ConventionalNEAT (const std::unique_ptr<Genome<Args...>>& genome, double a, double b, double c);
		double Euclidian (const std::unique_ptr<Genome<Args...>>& genome);

	template <typename... Args2>
	friend class Population;
};

}


/* IMPLEMENTATION */

using namespace pneatm;

template <typename... Args>
Species<Args...>::Species(unsigned int id, const std::unordered_map<unsigned int, Connection>& connections, distanceFn dstType): 
	id (id),
	dstType (dstType),
	connections (connections),
	avgFitness (0),
	avgFitnessAdjusted (0),
	allowedOffspring (0),
	sumFitness (0),
	gensSinceImproved (0),
	isDead (false) {
}

template <typename... Args>
Species<Args...>::Species (std::ifstream& inFile) {
	deserialize (inFile);
}


template <typename... Args>
double Species<Args...>::distanceWith (const std::unique_ptr<Genome<Args...>>& genome, double a, double b, double c) {
	switch (dstType) {
		case CONVENTIONAL:
			return ConventionalNEAT (genome, a, b, c);
			break;
		case EUCLIDIAN:
			return Euclidian (genome);
			break;
	}
	return 0.0;	// avoid compilation error
}


template <typename... Args>
double Species<Args...>::ConventionalNEAT (const std::unique_ptr<Genome<Args...>>& genome, double a, double b, double c) {
	// get enabled connections and maxInnovId for genome 1
	unsigned int maxInnovId1 = 0;
	std::vector<unsigned int> connEnabled1;
	for (const std::pair<const unsigned int, Connection>& conn : genome->connections) {
		if (conn.second.enabled) {
			connEnabled1.push_back (conn.second.id);
			if (conn.second.innovId > maxInnovId1) {
				maxInnovId1 = conn.second.innovId;
			}
		}
	}

	// get enabled connections and maxInnovId for genome 2
	unsigned int maxInnovId2 = 0;
	std::vector<unsigned int> connEnabled2;
	for (const std::pair<const unsigned int, Connection>& conn : connections) {
		if (conn.second.enabled) {
			connEnabled2.push_back (conn.second.id);
			if (conn.second.innovId > maxInnovId2) {
				maxInnovId2 = conn.second.innovId;
			}
		}
	}

	unsigned int excessGenes = 0;
	unsigned int disjointGenes = 0;
	double sumDiffWeights = 0.0;
	unsigned int nbCommonGenes = 0;

	for (size_t i1 = 0; i1 < connEnabled1.size (); i1++) {
		// for each enabled connection of the first genome
		if (genome->connections [connEnabled1 [i1]].innovId > maxInnovId2) {
			// if connection's innovId is over the maximum one of second genome's connections
			// it is an excess gene
			excessGenes += 1;
		} else {
			size_t i2 = 0;

			while (i2 < connEnabled2.size () && connections [connEnabled2 [i2]].innovId != genome->connections [connEnabled1 [i1]].innovId) {
				i2 ++;
			}
			if (i2 == connEnabled2.size ()) {
				// no connection with the same innovation id have been found in the second genome
				// it is a disjoint gene
				disjointGenes += 1;
			} else {
				// one connection has the same innovation id
				nbCommonGenes += 1;
				double diff = connections [connEnabled2 [i2]].weight - genome->connections [connEnabled1 [i1]].weight;
				if (diff > 0) {
					sumDiffWeights += diff;
				} else {
					sumDiffWeights += -1 * diff;
				}
			}
		}
	}

	for (size_t i2 = 0; i2 < connEnabled2.size (); i2++) {
		// for each enabled connection of the second genome
		if (connections [connEnabled2 [i2]].innovId > maxInnovId1) {
			// if connection's innovId is over the maximum one of first genome's connections
			// it is an excess gene
			excessGenes += 1;
		} else {
			size_t i1 = 0;
			while (i1 < connEnabled1.size () && connections [connEnabled2 [i2]].innovId != genome->connections [connEnabled1 [i1]].innovId) {
				i1 ++;
			}
			if (i1 == connEnabled1.size ()) {
				// no connection with the same innovation id have been found in the first genome
				// it is a disjoint gene
				disjointGenes += 1;
			}	// else, the weight's difference has already been processed in the previous for loop
		}
	}

	if (nbCommonGenes > 0) {
		return (
			(a * (double) excessGenes + b * (double) disjointGenes) / (double) std::max (connEnabled1.size (), connEnabled2.size ())
			+ c * sumDiffWeights / (double) nbCommonGenes
		);
	} else {
		// there is no common genes between genomes
		// let's return the maximum double as they might be very differents
		return std::numeric_limits<double>::max ();
	}
}

template <typename... Args>
double Species<Args...>::Euclidian (const std::unique_ptr<Genome<Args...>>& genome) {
	double result = 0.0;

	std::vector<size_t> usedId;
	for (const std::pair<const unsigned int, Connection>& conn : connections) {
		size_t i = 0;
		while (i < genome->connections.size () && genome->connections [(unsigned int) i].innovId != conn.second.innovId) {
			i++;
		}
		if (i >= genome->connections.size ()) {
			// the connection is not in the genome
			result += conn.second.weight * conn.second.weight;
		} else {
			usedId.push_back (i);
			// the leader and the genome share this connection
			result += (conn.second.weight - genome->connections [(unsigned int) i].weight) * (conn.second.weight - genome->connections [(unsigned int) i].weight);
		}
	}

	for (size_t i = 0; i < genome->connections.size (); i++) {
		size_t k = 0;
		while (k < usedId.size () && usedId [k] != i) {
			k++;
		}
		if (k >= usedId.size ()) {
			// the connection has not been take into account
			result += genome->connections [(unsigned int) i].weight * genome->connections [(unsigned int) i].weight;
		}
	}

	return result;	// actualy the euclidian distance is equal to the squared root of result: but this has no effect as we are comparing values
}

template <typename... Args>
void Species<Args...>::print (const std::string& prefix) const {
	std::cout << prefix << "ID: " << id << std::endl;
	std::cout << prefix << "Current Average Fitness: " << avgFitness << std::endl;
	std::cout << prefix << "Current Average Fitness Adjusted: " << avgFitnessAdjusted << std::endl;
	std::cout << prefix << "Current Number of Allowed Offspring: " << allowedOffspring << std::endl;
	std::cout << prefix << "Current Fitness Sum: " << sumFitness << std::endl;
	std::cout << prefix << "Current Number of generation since any improvement: " << gensSinceImproved << std::endl;
	std::cout << prefix << "Is dead? " << isDead << std::endl;
	std::cout << prefix << "Members' IDs: ";
	for (unsigned int id : members) {
		std::cout << id << ", ";
	}
	std::cout << std::endl;
}

template <typename... Args>
void Species<Args...>::serialize (std::ofstream& outFile) const {
	Serialize (id, outFile);
	Serialize (dstType, outFile);

	Serialize (connections.size (), outFile);
	for (const std::pair<const unsigned int, Connection>& conn : connections) {
		conn.second.serialize (outFile);
	}

	Serialize (avgFitness, outFile);
	Serialize (avgFitnessAdjusted, outFile);
	Serialize (allowedOffspring, outFile);
	Serialize (sumFitness, outFile);
	Serialize (gensSinceImproved, outFile);
	Serialize (isDead, outFile);
	Serialize (members, outFile);
}

template <typename... Args>
void Species<Args...>::deserialize (std::ifstream& inFile) {
	Deserialize (id, inFile);
	Deserialize (dstType, inFile);

	size_t sz;

	Deserialize (sz, inFile);
	connections.clear ();
	connections.reserve (sz);
	for (unsigned int k = 0; k < (unsigned int) sz; k++) {
		connections.insert (std::make_pair (k, Connection (inFile)));
	}

	Deserialize (avgFitness, inFile);
	Deserialize (avgFitnessAdjusted, inFile);
	Deserialize (allowedOffspring, inFile);
	Deserialize (sumFitness, inFile);
	Deserialize (gensSinceImproved, inFile);
	Deserialize (isDead, inFile);
	Deserialize (members, inFile);
}


#endif	// SPECIES_HPP