#include <setup.hpp>

int main () {
    srand ((int) time (0));	// init seed for rand

    // init logger
	spdlog::set_pattern ("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::err);
    auto logger = spdlog::stdout_color_mt("logger");

    unsigned int popSize = 100;
    std::vector<size_t> bias_sch = {1, 1};
    std::vector<size_t> inputs_sch = {14, 1};
    std::vector<size_t> outputs_sch = {1, 0};
    std::vector<std::vector<size_t>> hiddens_sch_init = {{3, 1}, {1, 1}};
    std::vector<void*> bias_init;
    float unitValueFLOAT = 1.0f;
    myInt unitValueINT (1);
    bias_init.push_back ((void*) &unitValueINT);
    bias_init.push_back ((void*) &unitValueFLOAT);
    std::vector<void*> resetValues;
    float nullValueFLOAT = 0.0f;
    myInt nullValueINT (0);
    resetValues.push_back ((void*) &nullValueINT);
    resetValues.push_back ((void*) &nullValueFLOAT);
    std::vector<std::vector<std::vector<void*>>> activationFns;
    activationFns.push_back ({});
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0].push_back ({});
    activationFns [1].push_back ({});
    activationFns [1].push_back ({});
    activationFns [1][1].push_back ((void*) &sigmoid_float2float);
    activationFns [0][0].push_back ((void*) &sigmoid_int2int);
    activationFns [1][0].push_back ((void*) &sigmoid_float2int);
    activationFns [0][1].push_back ((void*) &sigmoid_int2float);
    unsigned int N_ConnInit = 40;
    float probRecuInit = 0.0f;
    float weightExtremumInit = 20.0f;
    unsigned int maxRecuInit = 0;
    float speciationThreshInit = 100.0f;
    unsigned int threshGensSinceImproved = 15;
    pneatm::Population<myInt, float> pop (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger.get (), speciationThreshInit, threshGensSinceImproved);

    Snake snake (8);

    unsigned int maxRecurrency = 0;
    float mutateWeightThresh = 0.2f;
    float mutateWeightFullChangeThresh = 0.1f;
    float mutateWeightFactor = 1.2f;
    float addConnectionThresh = 0.15f;
    unsigned int maxIterationsFindConnectionThresh = 100;
    float reactivateConnectionThresh = 0.3f;
    float addNodeThresh = 0.2f;
    unsigned int maxIterationsFindNodeThresh = 200;
    float addTranstypeThresh = 0.2f;

    unsigned int maxIterationThresh = 500;
    float bestFitness = 0.0f;
    while (bestFitness < 2000.0f && pop.getGeneration () < 10000) {
        std::cout << "generation " << pop.getGeneration () << std::endl;

        for (unsigned int genomeId = 0; genomeId < popSize; genomeId ++) {
            snake.reset ();

            bool isFinished = false;
            unsigned int iteration = 0;
            while (iteration < maxIterationThresh && !isFinished) {
                std::vector<myInt> AI_Inputs = snake.getAIInputs ();

                for (unsigned int i = 0; i < 14; i++) {
                    pop.template loadInput<myInt> (AI_Inputs [i], i, genomeId);
                }
                pop.template loadInput<float> (snake.getScore (), 14, genomeId);

                pop.runNetwork (genomeId);

                myInt Snake_Inputs = pop.template getOutput<myInt> (0, genomeId);

                isFinished = snake.run (Snake_Inputs);

                iteration ++;
            }

            pop.setFitness (snake.getScore (), genomeId);
        }

        pop.speciate ();
        bestFitness = pop.getFitterGenome ().getFitness ();
        std::cout << "  - best fitness: " << bestFitness << std::endl;
        pop.crossover (true);
        pop.mutate (maxRecurrency, mutateWeightThresh, mutateWeightFullChangeThresh, mutateWeightFactor, addConnectionThresh, maxIterationsFindConnectionThresh, reactivateConnectionThresh, addNodeThresh, maxIterationsFindNodeThresh, addTranstypeThresh);
    }

    // we have to run once again the network and to do a speciation to get the last fitter genome
    for (unsigned int genomeId = 0; genomeId < popSize; genomeId ++) {
        snake.reset ();

        bool isFinished = false;
        unsigned int iteration = 0;
        while (iteration < maxIterationThresh && !isFinished) {
            std::vector<myInt> AI_Inputs = snake.getAIInputs ();

            for (unsigned int i = 0; i < 14; i++) {
                pop.template loadInput<myInt> (AI_Inputs [i], i, genomeId);
            }
            pop.template loadInput<float> (snake.getScore (), 14, genomeId);

            pop.runNetwork (genomeId);

            myInt Snake_Inputs = pop.template getOutput<myInt> (0, genomeId);

            isFinished = snake.run (Snake_Inputs);

            iteration ++;
        }
        if (!isFinished) {
            pop.setFitness (0.0f, genomeId);
        } else {
            pop.setFitness (snake.getScore (), genomeId);
        }
    }
    pop.speciate ();

    pop.getFitterGenome ().print ();
    pop.getFitterGenome ().draw ();

    // play a game by the fitter genome
    playGameFitter (pop.getFitterGenome (), maxIterationThresh, false, {800, 600}, 0.12f, 8);

    return 0;
}
