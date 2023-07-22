#include <setup.hpp>

int main () {
    srand ((int) time (0));	// init seed for rand function

    // init pneatm logger
	spdlog::set_pattern ("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::err);
    auto logger = spdlog::rotating_logger_mt("pneatm_logger", "logs/log.txt", 1048576 * 100, 500);

    // init stats logger

    // init population
    unsigned int popSize = 100;
    pneatm::Population<myInt, myFloat> pop = SetupPopulation (popSize, logger.get ());

    // init snake
    Snake snake (8);

    // init mutation parameters
    std::function<pneatm::mutationParams_t (double)> paramsMap = SetupMutationParametersMaps ();

    unsigned int maxIterationThresh = 500;
    int nbGame = 5;
    double bestFitness = 0.0;
    while (bestFitness < 2000.0 && pop.getGeneration () < 150) { // while goal is not reach
        std::cout << "generation " << pop.getGeneration () << std::endl;

        for (unsigned int genomeId = 0; genomeId < popSize; genomeId ++) {

            float score = 0.0f;
            for (int g = 0; g < nbGame; g++) {
                snake.reset ();

                bool isFinished = false;
                unsigned int iteration = 0;
                while (iteration < maxIterationThresh && !isFinished) { // while game has not ended
                    // get inputs from snake's eyes
                    std::vector<myInt> AI_Inputs = snake.getAIInputs ();

                    // load inputs in the genome's network
                    for (unsigned int i = 0; i < 14; i++) {
                        pop.template loadInput<myInt> (AI_Inputs [i], i, genomeId);
                    }
                    pop.template loadInput<myFloat> (snake.getScore (), 14, genomeId);

                    // run the network
                    pop.runNetwork (genomeId);

                    // get output, the movement order to give to the snake
                    myInt Snake_Inputs = pop.template getOutput<myInt> (0, genomeId);

                    // move the snake
                    isFinished = snake.run (Snake_Inputs);

                    iteration ++;
                }

                score += snake.getScore ();

                pop.resetMemory (genomeId);
            }

            // game has ended, we set the score to the genome's fitness
            pop.setFitness (score / (float) nbGame, genomeId);
        }

        // speciation step
        pop.speciate (5, 0, 100, 0.3);

        bestFitness = pop.getGenome ().getFitness ();
        std::cout << "  - best fitness: " << bestFitness << std::endl;

        // crossover step
        pop.crossover (true, 0.8);

        // mutation step
        pop.mutate (paramsMap);
    }

    // we have to run once again the network to do a speciation to get the last fitter genome
    for (unsigned int genomeId = 0; genomeId < popSize; genomeId ++) {
        float score = 0.0f;
        for (int g = 0; g < nbGame; g++) {
            snake.reset ();

            bool isFinished = false;
            unsigned int iteration = 0;
            while (iteration < maxIterationThresh && !isFinished) { // while game has not ended
                // get inputs from snake's eyes
                std::vector<myInt> AI_Inputs = snake.getAIInputs ();

                // load inputs in the genome's network
                for (unsigned int i = 0; i < 14; i++) {
                    pop.template loadInput<myInt> (AI_Inputs [i], i, genomeId);
                }
                pop.template loadInput<myFloat> (snake.getScore (), 14, genomeId);

                // run the network
                pop.runNetwork (genomeId);

                // get output, the movement order to give to the snake
                myInt Snake_Inputs = pop.template getOutput<myInt> (0, genomeId);

                // move the snake
                isFinished = snake.run (Snake_Inputs);

                iteration ++;
            }

            score += snake.getScore ();

            pop.resetMemory (genomeId);
        }

        // game has ended, we set the score to the genome's fitness
        pop.setFitness (score / (float) nbGame, genomeId);
    }
    pop.speciate (10, 0, 100, 0.3);

    // print info and draw genome's network
    pop.getGenome (-1).print ();
    pop.getGenome (-1).draw ("/usr/share/fonts/OTF/SF-Pro-Rounded-Light.otf");

    // play a game by the fitter genome
    playGameFitter (pop.getGenome (-1), maxIterationThresh, false, {800, 600}, 0.12f, 8);

    return 0;
}
