#ifndef SETUP_HPP
#define SETUP_HPP

#include <iostream>
#include <vector>
#include <snake.hpp>
#include <PNEATM/population.hpp>
#include <PNEATM/genome.hpp>
#include <functional>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <myInt.hpp>

/* SNAKE */

template <typename... Args>
void playGameFitter (pneatm::Genome<Args...>& genome, const unsigned int maxIterationThresh, bool displayConsole = true, sf::Vector2u windowSize = {800, 600}, float timeUpsSeconds = 0.7f, const unsigned int playgroundSize = 8) {
    sf::RenderWindow window(sf::VideoMode(windowSize.x, windowSize.y), "PNEATM - https://github.com/titofra");

    Snake snake (playgroundSize);

    if (displayConsole) {
        snake.drawPlaygroundConsole ();
    } else {
        snake.drawPlaygroundSFML (&window, timeUpsSeconds);
    }

    bool isFinished = false;
    unsigned int iteration = 0;
    while (iteration < maxIterationThresh && !isFinished) {
        std::vector<myInt> AI_Inputs = snake.getAIInputs ();

        for (unsigned int i = 0; i < 14; i++) {
            genome.template loadInput<myInt> (AI_Inputs [i], i);
        }
        genome.template loadInput<float> (snake.getScore (), 14);

        genome.runNetwork ();

        myInt Snake_Inputs = genome.template getOutput<myInt> (0);

        isFinished = snake.run (Snake_Inputs);

        if (displayConsole) {
            snake.drawPlaygroundConsole ();
        } else {
            snake.drawPlaygroundSFML (&window, timeUpsSeconds);
        }

        iteration ++;
    }
	if (displayConsole) {
        snake.drawPlaygroundConsole ();
    } else {
        snake.drawPlaygroundSFML (&window, timeUpsSeconds);
    }

	std::cout << "final score: " << snake.getScore () << std::endl;
}


/* ACTIVATION FUNCTIONS */

std::function<float (float)> sigmoid_float2float = [] (float x) {
    return (1.0f / (1.0f + (float) exp(-1 * 4.09 * (double) x)));
};

std::function<myInt (myInt)> sigmoid_int2int = [] (myInt x) {
    return myInt ((int) (1.0 / (1.0 + exp(-1 * 4.09 * (double) x))));
};

std::function<float (myInt)> sigmoid_int2float = [] (myInt x) {
    return (1.0f / (1.0f + (float) exp(-1 * 4.09 * (double) x)));
};

std::function<myInt (float)> sigmoid_float2int = [] (float x) {
    return myInt ((int) (1.0f / (1.0f + (float) exp(-1 * 4.09 * (double) x))));
};


/* SETUP FUNCTIONS */

pneatm::Population<myInt, float> SetupPopulation (unsigned int popSize, spdlog::logger* logger) {
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
    float speciationThreshInit = 20.0f;
    unsigned int threshGensSinceImproved = 15;
    return pneatm::Population<myInt, float> (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger, speciationThreshInit, threshGensSinceImproved, "stats16.csv");
}

std::function<pneatm::mutationParams_t (float)> SetupMutationParametersMaps () {
    pneatm::mutationParams_t explorationSet;
    explorationSet.nodes.rate = 0.25f;
    explorationSet.nodes.monotypedRate = 0.5f;
    explorationSet.nodes.monotyped.maxIterationsFindConnection = 100;
    explorationSet.nodes.bityped.maxRecurrencyEntryConnection = 0;
    explorationSet.nodes.bityped.maxIterationsFindNode = 100;
    explorationSet.connections.rate = 0.25f;
    explorationSet.connections.reactivateRate = 0.3f;
    explorationSet.connections.maxRecurrency = 0;
    explorationSet.connections.maxIterations = 100;
    explorationSet.connections.maxIterationsFindNode = 100;
    explorationSet.weights.rate = 0.15f;
    explorationSet.weights.fullChangeRate = 0.3f;
    explorationSet.weights.perturbationFactor = 2.0f;
    pneatm::mutationParams_t refinementSet;
    refinementSet.nodes.rate = 0.1f;
    refinementSet.nodes.monotypedRate = 0.5f;
    refinementSet.nodes.monotyped.maxIterationsFindConnection = 100;
    refinementSet.nodes.bityped.maxRecurrencyEntryConnection = 0;
    refinementSet.nodes.bityped.maxIterationsFindNode = 100;
    refinementSet.connections.rate = 0.1f;
    refinementSet.connections.reactivateRate = 0.2f;
    refinementSet.connections.maxRecurrency = 0;
    refinementSet.connections.maxIterations = 100;
    refinementSet.connections.maxIterationsFindNode = 100;
    refinementSet.weights.rate = 0.05f;
    refinementSet.weights.fullChangeRate = 0.1f;
    refinementSet.weights.perturbationFactor = 1.2f;
    return [=] (float fitness) {
        // Here, the mutation map is very basic: if the genome is pretty good, we just refine his network, else we explore new networks
        if (fitness > 800.0f) {
            return refinementSet;
        }
        return explorationSet;
    };
}

#endif  // SETUP_HPP