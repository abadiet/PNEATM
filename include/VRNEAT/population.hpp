#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <LRNEAT/genome.hpp>
#include <LRNEAT/species.hpp>
#include <LRNEAT/activation_fn.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>

namespace neat {

class Population : Genome{
	public:
		Population (int popSize, std::vector<int> nbInput, std::vector<int> nbOutput, std::vector<int> nbHiddenInit, int bias_kind, void* bias_kind, float probConnInit, bool areRecurrentConnectionsAllowed = false, float weightExtremumInit = 20.0f, float speciationThreshInit = 100.0f, int threshGensSinceImproved = 15);
		Population (const std::string filepath) {load(filepath);};

		void addKind (int kind);
		void addActivationFn (void* fn(void* input), int arg_kind, int out_kind);
		void loadInputs (void* inputs []);
		void loadInput (void* input, int input_id);
		void loadInputs (void* inputs [], int genome_id);
		void loadInput (void* input, int input_id, int genome_id);
		void runNetwork ();
		void runNetwork (int genome_id);
		void getOutputs (void* outputs [], int genome_id);
		void* getOutput (int output_id, int genome_id);
		void setFitness (float fitness, int genome_id);
		void speciate (int target = 5, int targetThresh = 0, float stepThresh = 0.5f, float a = 1.0f, float b = 1.0f, float c = 0.4f);
		void crossover (bool elitism = false);
		void mutate (float mutateWeightThresh = 0.8f, float mutateWeightFullChangeThresh = 0.1f, float mutateWeightFactor = 1.2f, float addConnectionThresh = 0.05f, int maxIterationsFindConnectionThresh = 20, float reactivateConnectionThresh = 0.25f, float addNodeThresh = 0.03f, int maxIterationsFindNodeThresh = 20);
		void drawNetwork (int genome_id, sf::Vector2u windowSize = {1300, 800}, float dotsRadius = 6.5f);
		void printInfo (bool extendedGlobal = false, bool printSpecies = false, bool printGenomes = false, bool extendedGenomes = false);
		void save (const std::string filepath = "./neat_backup.txt");
		void load (const std::string filepath = "./neat_backup.txt");
	
	private:
		int generation;
		float avgFitness;
		float avgFitnessAdjusted;

		std::vector<std::vector<int>> innovIds;
		int lastInnovId;

		int popSize;
		float speciationThresh;
		int threshGensSinceImproved;
		std::vector<int> nbInput;	// only useful for creating new genome
		std::vector<int> nbOutput;	// only useful for creating new genome
		std::vector<int> nbHiddenInit;	// only useful for creating new genome
		float probConnInit;	// only useful for creating new genome
		int bias_kind;	// only useful for creating new genome
		void* bias_init;	// only useful for creating new genome
		float weightExtremumInit;	// only useful for creating new genome
		bool areRecurrentConnectionsAllowed;
		int fittergenome_id;
		std::vector<Genome> genomes;
		std::vector<Species> species;
		std::vector<int> kinds;
		std::vector<ActivationFn> activationFns;

		float compareGenomes(int ig1, int ig2, float a, float b, float c);
		void updateFitnesses();
		int selectParent(int iSpe);
};

}

#endif	// POPULATION_HPP
