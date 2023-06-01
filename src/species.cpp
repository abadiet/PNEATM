#include <VRNEAT/species.hpp>

using namespace vrneat;

Species::Species(int id): id(id) {
	avgFitness = 0;
	avgFitnessAdjusted = 0;
	sumFitness = 0;
	gensSinceImproved = 0;
	isDead = false;
}
