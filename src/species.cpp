#include <PNEATM/species.hpp>

using namespace pneatm;

Species::Species(unsigned int id): id(id) {
	avgFitness = 0;
	avgFitnessAdjusted = 0;
	sumFitness = 0;
	gensSinceImproved = 0;
	isDead = false;
}
