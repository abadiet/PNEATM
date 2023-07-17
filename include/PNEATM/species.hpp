#ifndef SPECIES_HPP
#define SPECIES_HPP

#include <PNEATM/genome.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <cstring>

/* HEADER */

namespace pneatm {

template <typename... Args>
class Species {
	public:
		Species (unsigned int id, const std::unique_ptr<Genome<Args...>>& leader);
		Species(const Species<Args...>& other);
		~Species () {};

		Species<Args...>& operator= (const Species& other);

		float distanceWith (const std::unique_ptr<Genome<Args...>>& genome, float a, float b, float c);

		void print (std::string prefix = "");

	private:
		unsigned int id;
		std::unique_ptr<Genome<Args...>> leader;
		float avgFitness;
		float avgFitnessAdjusted;
		int allowedOffspring;
		float sumFitness;
		unsigned int gensSinceImproved;
		bool isDead;
		std::vector<unsigned int> members;

	template <typename... Args2>
	friend class Population;
};

}


/* IMPLEMENTATION */

using namespace pneatm;

template <typename... Args>
Species<Args...>::Species(unsigned int id, const std::unique_ptr<Genome<Args...>>& leader): 
	id (id),
	leader (leader->clone ()),
	avgFitness (0),
	avgFitnessAdjusted (0),
	allowedOffspring (0),
	sumFitness (0),
	gensSinceImproved (0),
	isDead (false) {
}

template <typename... Args>
Species<Args...>::Species (const Species<Args...>& other) :
    id (other.id),
    leader (other.leader->clone ()),
    avgFitness (other.avgFitness),
    avgFitnessAdjusted (other.avgFitnessAdjusted),
    allowedOffspring (other.allowedOffspring),
    sumFitness (other.sumFitness),
    gensSinceImproved (other.gensSinceImproved),
    isDead (other.isDead),
    members (other.members) {
}

template <typename... Args>
Species<Args...>& Species<Args...>::operator= (const Species& other) {
    if (this != &other) {
        id = other.id;
        leader = other.leader->clone();
        avgFitness = other.avgFitness;
        avgFitnessAdjusted = other.avgFitnessAdjusted;
        allowedOffspring = other.allowedOffspring;
        sumFitness = other.sumFitness;
        gensSinceImproved = other.gensSinceImproved;
        isDead = other.isDead;
        members = other.members;
    }
    return *this;
}

template <typename... Args>
float Species<Args...>::distanceWith (const std::unique_ptr<Genome<Args...>>& genome, float a, float b, float c) {
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
	for (size_t i = 0; i < leader->connections.size (); i++) {
		if (leader->connections [i].enabled) {
			connEnabled2.push_back (i);
			if (leader->connections [i].innovId > maxInnovId2) {
				maxInnovId2 = leader->connections [i].innovId;
			}
		}
	}

	unsigned int excessGenes = 0;
	unsigned int disjointGenes = 0;
	float sumDiffWeights = 0.0f;
	unsigned int nbCommonGenes = 0;

	for (size_t i1 = 0; i1 < connEnabled1.size (); i1++) {
		// for each enabled connection of the first genome
		if (genome->connections [connEnabled1 [i1]].innovId > maxInnovId2) {
			// if connection's innovId is over the maximum one of second genome's connections
			// it is an excess gene
			excessGenes += 1;
		} else {
			size_t i2 = 0;

			while (i2 < connEnabled2.size () && leader->connections [connEnabled2 [i2]].innovId != genome->connections [connEnabled1 [i1]].innovId) {
				i2 ++;
			}
			if (i2 == connEnabled2.size ()) {
				// no connection with the same innovation id have been found in the second genome
				// it is a disjoint gene
				disjointGenes += 1;
			} else {
				// one connection has the same innovation id
				nbCommonGenes += 1;
				float diff = leader->connections [connEnabled2 [i2]].weight - genome->connections [connEnabled1 [i1]].weight;
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
		if (leader->connections [connEnabled2 [i2]].innovId > maxInnovId1) {
			// if connection's innovId is over the maximum one of first genome's connections
			// it is an excess gene
			excessGenes += 1;
		} else {
			size_t i1 = 0;
			while (i1 < connEnabled1.size () && leader->connections [connEnabled2 [i2]].innovId != genome->connections [connEnabled1 [i1]].innovId) {
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
			(a * (float) excessGenes + b * (float) disjointGenes) / (float) std::max (connEnabled1.size (), connEnabled2.size ())
			+ c * sumDiffWeights / (float) nbCommonGenes
		);
	} else {
		// there is no common genes between genomes
		// let's return the maximum float as they might be very differents
		return std::numeric_limits<float>::max ();
	}
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