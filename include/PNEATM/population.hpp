#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <PNEATM/genome.hpp>
#include <PNEATM/species.hpp>
#include <PNEATM/utils.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>

namespace pneatm {

template <typename... Args>
class Population {
	public:
		Population (unsigned int popSize, std::vector<size_t> bias_sch, std::vector<size_t> inputs_sch, std::vector<size_t> outputs_sch, std::vector<std::vector<size_t>> hiddens_sch_init, std::vector<void*> bias_init, std::vector<void*> resetValues, std::vector<std::vector<std::vector<std::function <void* (void*)>>>> activationFns, unsigned int N_ConnInit, float probRecuInit, float weightExtremumInit, unsigned int maxRecuInit, float speciationThreshInit = 100.0f, int threshGensSinceImproved = 15);
		//Population (const std::string filepath) {load(filepath);};

		template <typename T_in>
		void loadInputs (T_in inputs []);
		template <typename T_in>
		void loadInput (T_in input, unsigned int input_id);
		template <typename T_in>
		void loadInputs (T_in inputs [], unsigned int genome_id);
		template <typename T_in>
		void loadInput (T_in input, unsigned int input_id, unsigned int genome_id);

		void runNetwork ();
		void runNetwork (unsigned int genome_id);

		template <typename T_out>
		void getOutputs (T_out outputs [], unsigned int genome_id);
		template <typename T_out>
		T_out getOutput (unsigned int output_id, unsigned int genome_id);

		void setFitness (float fitness, unsigned int genome_id);
		void speciate (unsigned int target = 5, unsigned int targetThresh = 0, float stepThresh = 0.5f, float a = 1.0f, float b = 1.0f, float c = 0.4f);
		void crossover (bool elitism = false);
		void mutate (unsigned int maxRecurrency, float mutateWeightThresh = 0.8f, float mutateWeightFullChangeThresh = 0.1f, float mutateWeightFactor = 1.2f, float addConnectionThresh = 0.05f, unsigned int maxIterationsFindConnectionThresh = 20, float reactivateConnectionThresh = 0.25f, float addNodeThresh = 0.03f, unsigned int maxIterationsFindNodeThresh = 20, float addTranstypeThresh = 0.02f);
		/*void drawNetwork (unsigned int genome_id, sf::Vector2u windowSize = {1300, 800}, float dotsRadius = 6.5f);
		void printInfo (bool extendedGlobal = false, bool printSpecies = false, bool printGenomes = false, bool extendedGenomes = false);
		void save (const std::string filepath = "./neat_backup.txt");
		void load (const std::string filepath = "./neat_backup.txt");*/
	
	private:
		unsigned int generation;
		float avgFitness;
		float avgFitnessAdjusted;
		unsigned int popSize;
		float speciationThresh;
		unsigned int threshGensSinceImproved;

		// useful parameters to create new genome
		std::vector<size_t> bias_sch;
		std::vector<size_t> inputs_sch;
		std::vector<size_t> outputs_sch;
		std::vector<std::vector<size_t>> hiddens_sch_init;
		std::vector<void*> bias_init;
		std::vector<void*> resetValues;
		unsigned int N_ConnInit;
		float probRecuInit;
		float weightExtremumInit;
		unsigned int maxRecuInit;

		int fittergenome_id;
		std::vector<Genome<Args...>> genomes;
		std::vector<Species> species;
		std::vector<std::vector<std::vector<std::function <void* (void*)>>>> activationFns;
		innovation_t conn_innov;

		float CompareGenomes (unsigned int ig1, unsigned int ig2, float a, float b, float c);
		void UpdateFitnesses ();
		int SelectParent (unsigned int iSpe);
};

}

#endif	// POPULATION_HPP
