#ifndef SPECIES_HPP
#define SPECIES_HPP

#include <vector>

namespace pneatm {

class Species{
	public:
		Species(int id);

	protected:
		float avgFitness;
		float avgFitnessAdjusted;
		int allowedOffspring;
		float sumFitness;
		int gensSinceImproved;
		bool isDead;
		std::vector<int> members;

	private:
		int id;
};

}

#endif	// SPECIES_HPP