#ifndef SPECIES_HPP
#define SPECIES_HPP

#include <PNEATM/Connection/connection.hpp>
#include <PNEATM/genome.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <cstring>

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
		 * @param connections A vector of connection that define the species traits. Will be used to process the distance between the species and genomes.
		 * @param dstType The distance algorithm to use:\n	- CONVENTIONAL: algorithm used in the original NEAT\n	- EUCLIDIAN: euclidian distance in the connections's space
		 */
		Species (unsigned int id, std::vector<Connection> connections, distanceFn dstType);

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
		void print (std::string prefix = "");

	private:
		unsigned int id;
		distanceFn dstType;
		std::vector<Connection> connections;
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
Species<Args...>::Species(unsigned int id, std::vector<Connection> connections, distanceFn dstType): 
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
	std::vector<size_t> connEnabled1;
	for (size_t i = 0; i < genome->connections.size (); i++) {
		if (genome->connections [i].enabled) {
			connEnabled1.push_back(i);
			if (genome->connections [i].innovId > maxInnovId1) {
				maxInnovId1 = genome->connections [i].innovId;
			}
		}
	}

	// get enabled connections and maxInnovId for genome 2
	unsigned int maxInnovId2 = 0;
	std::vector<size_t> connEnabled2;
	for (size_t i = 0; i < connections.size (); i++) {
		if (connections [i].enabled) {
			connEnabled2.push_back (i);
			if (connections [i].innovId > maxInnovId2) {
				maxInnovId2 = connections [i].innovId;
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
	for (const Connection& conn : connections) {
		size_t i = 0;
		while (i < genome->connections.size () && genome->connections [i].innovId != conn.innovId) {
			i++;
		}
		if (i >= genome->connections.size ()) {
			// the connection is not in the genome
			result += conn.weight * conn.weight;
		} else {
			usedId.push_back (i);
			// the leader and the genome share this connection
			result += (conn.weight - genome->connections [i].weight) * (conn.weight - genome->connections [i].weight);
		}
	}

	for (size_t i = 0; i < genome->connections.size (); i++) {
		size_t k = 0;
		while (k < usedId.size () && usedId [k] != i) {
			k++;
		}
		if (k >= usedId.size ()) {
			// the connection has not been take into account
			result += genome->connections [i].weight * genome->connections [i].weight;
		}
	}

	return result;	// actualy the euclidian distance is equal to the squared root of result: but this has no effect as we are comparing values
}

template <typename... Args>
void Species<Args...>::print (std::string prefix) {
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

#endif	// SPECIES_HPP