#ifndef SPECIES_HPP
#define SPECIES_HPP

#include <vector>

namespace pneatm {

class Species {
	public:
		Species (unsigned int id);

	private:
		unsigned int id;
		float avgFitness;
		float avgFitnessAdjusted;
		int allowedOffspring;
		float sumFitness;
		unsigned int gensSinceImproved;
		bool isDead;
		std::vector<int> members;

	template <typename... Args>
	friend class Population;
};

}

#endif	// SPECIES_HPP