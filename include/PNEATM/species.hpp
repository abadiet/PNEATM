#ifndef SPECIES_HPP
#define SPECIES_HPP

#include <vector>
#include <iostream>
#include <cstring>

namespace pneatm {

class Species {
	public:
		Species (unsigned int id);

		void print (std::string prefix = "");

	private:
		unsigned int id;
		float avgFitness;
		float avgFitnessAdjusted;
		int allowedOffspring;
		float sumFitness;
		unsigned int gensSinceImproved;
		bool isDead;
		std::vector<unsigned int> members;

	template <typename... Args>
	friend class Population;
};

}

#endif	// SPECIES_HPP