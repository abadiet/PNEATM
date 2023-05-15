#ifndef SPECIES_HPP
#define SPECIES_HPP

#include <vector>

namespace neat {

class Species{
	public:
		Species(int id);

	private:
		int id;
		float avgFitness;
		float avgFitnessAdjusted;
		int allowedOffspring;
		float sumFitness;
		int gensSinceImproved;
		bool isDead;
		std::vector<int> members;
};

}

#endif	// SPECIES_HPP