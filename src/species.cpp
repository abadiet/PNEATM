#include <PNEATM/species.hpp>

using namespace pneatm;

Species::Species(unsigned int id): id(id) {
	avgFitness = 0;
	avgFitnessAdjusted = 0;
	sumFitness = 0;
	gensSinceImproved = 0;
	isDead = false;
}

void Species::print (std::string prefix) {
	std::cout << prefix << "ID: " << id << std::endl;
	std::cout << prefix << "Current Average Fitness: " << avgFitness << std::endl;
	std::cout << prefix << "Current Average Fitness Adjusted: " << avgFitnessAdjusted << std::endl;
	std::cout << prefix << "Current Number of Allowed Offspring: " << allowedOffspring << std::endl;
	std::cout << prefix << "Current Fitness Sum: " << sumFitness << std::endl;
	std::cout << prefix << "Current Number of generation since any improvement: " << gensSinceImproved << std::endl;
	std::cout << prefix << "Is dead? " << isDead << std::endl;
	std::cout << prefix << "Members' IDs: ";
	for (int id : members) {
		std::cout << id << ", ";
	}
	std::cout << std::endl;
}
