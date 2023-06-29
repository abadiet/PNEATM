#include <setup.hpp>

int main () {
    srand ((int) time (0));	// init seed for rand

    // init logger
	spdlog::set_pattern ("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::err);
    auto logger = spdlog::stdout_color_mt("logger");

    unsigned int popSize = 50;
    std::vector<size_t> bias_sch = {1};
    std::vector<size_t> inputs_sch = {14};
    std::vector<size_t> outputs_sch = {3};
    std::vector<std::vector<size_t>> hiddens_sch_init = {{8}};
    std::vector<void*> bias_init;
    float unitValueFLOAT = 1.0f;
    bias_init.push_back ((void*) &unitValueFLOAT);
    std::vector<void*> resetValues;
    float nullValueFLOAT = 0.0f;
    resetValues.push_back ((void*) &nullValueFLOAT);
    std::vector<std::vector<std::vector<void*>>> activationFns;
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0][0].push_back ((void*) &sigmoid_float2float);
    unsigned int N_ConnInit = 60;
    float probRecuInit = 0.0f;
    float weightExtremumInit = 20.0f;
    unsigned int maxRecuInit = 0;
    float speciationThreshInit = 100.0f;
    int threshGensSinceImproved = 15;
    pneatm::Population<float> pop (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger.get (), speciationThreshInit, threshGensSinceImproved);

    unsigned int maxRecurrency = 0;

    Snake snake (8);

    unsigned int maxIterationThresh = 500;
    float bestFitness = 0.0f;
    while (bestFitness < 3000.0f && pop.getGeneration () < 10000) {
        std::cout << "generation " << pop.getGeneration () << std::endl;

        for (unsigned int genomeId = 0; genomeId < popSize; genomeId ++) {
            snake.reset ();

            std::vector<float> AI_Inputs;
            std::vector<float> Snake_Inputs;
            bool isFinished = false;
            unsigned int iteration = 0;
            while (iteration < maxIterationThresh && !isFinished) {
                AI_Inputs = snake.getAIInputs ();

                pop.template loadInputs<float> (AI_Inputs, genomeId);
                pop.runNetwork (genomeId);
                
                Snake_Inputs = pop.template getOutputs<float> (genomeId);

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
        bestFitness = pop.getFitterGenome ().getFitness ();
        std::cout << "  - best fitness: " << bestFitness << std::endl;
        pop.crossover ();
        pop.mutate (maxRecurrency);
    }

    // we have to run once again the network and to do a speciation to get the last fitter genome
    for (unsigned int genomeId = 0; genomeId < popSize; genomeId ++) {
        snake.reset ();

        std::vector<float> AI_Inputs;
        std::vector<float> Snake_Inputs;
        bool isFinished = false;
        unsigned int iteration = 0;
        while (iteration < maxIterationThresh && !isFinished) {
            AI_Inputs = snake.getAIInputs ();

            pop.template loadInputs<float> (AI_Inputs, genomeId);
            pop.runNetwork (genomeId);
            
            Snake_Inputs = pop.template getOutputs<float> (genomeId);

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

    // play a game by the fitter genome
    playGameFitter (pop.getFitterGenome (), maxIterationThresh, false, {800, 600}, 0.12f, 8);

    return 0;
}
