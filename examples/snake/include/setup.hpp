#ifndef SETUP_HPP
#define SETUP_HPP

#include <iostream>
#include <vector>
#include <snake.hpp>
#include <PNEATM/population.hpp>
#include <PNEATM/genome.hpp>
#include <PNEATM/species.hpp>
#include <PNEATM/Node/Activation_Function/activation_function.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <myTypes.hpp>

#define UNUSED(expr) do { (void) (expr); } while (0)

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
        float score = snake.getScore ();
        genome.template loadInput<float> (score, 14);

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

// parameters structure
typedef struct activationFnParams {
    // each parameters MUST HAVE a way to be initialised: it can be a constant value or like here a function (is called every time the network creates a new node)
    double alpha = Random_Double (-10.0, 10.0);
    double beta = Random_Double (-10.0, 10.0);
} activationFnParams_t;

// process functions (the activation functions)
std::function<myFloat (myFloat, activationFnParams_t*)> identity_float2float = [] (myFloat x, activationFnParams_t* params) -> myFloat {
    return x;
    UNUSED (params);
};
std::function<myInt (myInt, activationFnParams_t*)> identity_int2int = [] (myInt x, activationFnParams_t* params) -> myInt {
    return x;
    UNUSED (params);
};
std::function<myFloat (myFloat, activationFnParams_t*)> sigmoid_float2float = [] (myFloat x, activationFnParams_t* params) -> myFloat {
    return myFloat ((float) (1.0 / (1.0 + exp(-params->alpha * ((double) x - params->beta)))));
};
std::function<myInt (myInt, activationFnParams_t*)> sigmoid_int2int = [] (myInt x, activationFnParams_t* params) -> myInt {
    return myInt ((int) (1.0 / (1.0 + exp(-params->alpha * ((double) x - params->beta)))));
};
std::function<myFloat (myInt, activationFnParams_t*)> sigmoid_int2float = [] (myInt x, activationFnParams_t* params) -> myFloat {
    return myFloat ((float) (1.0 / (1.0 + exp(-params->alpha * ((double) x - params->beta)))));
};
std::function<myInt (myFloat, activationFnParams_t*)> sigmoid_float2int = [] (myFloat x, activationFnParams_t* params) -> myInt {
    return myInt ((int) (1.0 / (1.0 + exp(-params->alpha * ((double) x - params->beta)))));
};

// the printing function
std::function<void (activationFnParams_t*, std::string)> printingFn = [] (activationFnParams_t* params, std::string prefix) -> void {
    std::cout << "alpha = " << params->alpha << "   beta = " << params->beta << std::endl;
    UNUSED (prefix);
};

// the mutation function
std::function<void (activationFnParams_t*, double)> mutationFn = [] (activationFnParams_t* params, double fitness) -> void {
    // Here is the mutation function is very basic: if the genome is pretty good, we just refine his network, else we explore new networks
    if (fitness > 400.0) {
        if (Random_Double (0.0, 1.0, true, false) < 0.3) {
            // reset values
            params->alpha = Random_Double (-10.0, 10.0);
            params->beta = Random_Double (-10.0, 10.0);
        } else {
            // perturb values
            params->alpha += params->alpha * Random_Double (-0.2, 0.2);
            params->beta += params->beta * Random_Double (-0.2, 0.2);
        }
    } else {
        if (Random_Double (0.0, 1.0, true, false) < 0.4) {
            // reset values
            params->alpha = Random_Double (-10.0, 10.0);
            params->beta = Random_Double (-10.0, 10.0);
        } else {
            // perturb values
            params->alpha += params->alpha * Random_Double (-0.2, 0.2);
            params->beta += params->beta * Random_Double (-0.2, 0.2);
        }
    }
};


/* SETUP MAIN FUNCTIONS */

pneatm::Population<myInt, myFloat> SetupPopulation (unsigned int popSize, spdlog::logger* logger) {
    // nodes scheme setup
    std::vector<size_t> bias_sch = {1, 1};
    std::vector<size_t> inputs_sch = {14, 1};
    std::vector<size_t> outputs_sch = {1, 0};
    std::vector<std::vector<size_t>> hiddens_sch_init = {{3, 1}, {1, 1}};

    // bias values
    std::vector<void*> bias_init;
    myFloat* unitValueFLOAT = new myFloat (1.0f);
    myInt* unitValueINT = new myInt (1);
    bias_init.push_back ((void*) unitValueINT);
    bias_init.push_back ((void*) unitValueFLOAT);

    // reset values
    std::vector<void*> resetValues;
    myFloat* nullValueFLOAT = new myFloat (0.0f);
    myInt* nullValueINT = new myInt (0);
    resetValues.push_back ((void*) nullValueINT);
    resetValues.push_back ((void*) nullValueFLOAT);

    // activation functions setup
    std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
    activationFns.push_back ({});
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0].push_back ({});
    activationFns [1].push_back ({});
    activationFns [1].push_back ({});
    activationFns [0][0].push_back (new ActivationFn<myInt, myInt> ());
    activationFns [0][0].push_back (new ActivationFn<myInt, myInt> ());
    activationFns [1][1].push_back (new ActivationFn<myFloat, myFloat> ());
    activationFns [1][1].push_back (new ActivationFn<myFloat, myFloat> ());
    activationFns [0][1].push_back (new ActivationFn<myInt, myFloat> ());
    activationFns [1][0].push_back (new ActivationFn<myFloat, myInt> ());
    activationFns [0][0][0]->setFunction ((void*) &identity_int2int);   // identity function MUST BE the first
    activationFns [0][0][1]->setFunction ((void*) &sigmoid_int2int);
    activationFns [1][1][0]->setFunction ((void*) &identity_float2float);   // identity function MUST BE the first
    activationFns [1][1][1]->setFunction ((void*) &sigmoid_float2float);
    activationFns [0][1][0]->setFunction ((void*) &sigmoid_int2float);
    activationFns [1][0][0]->setFunction ((void*) &sigmoid_float2int);
    activationFns [0][0][0]->setPrintingFunction (printingFn);  // for all the activation functions, the printing and mutation functions are the sames
    activationFns [0][0][1]->setPrintingFunction (printingFn);
    activationFns [1][1][0]->setPrintingFunction (printingFn);
    activationFns [1][1][1]->setPrintingFunction (printingFn);
    activationFns [0][1][0]->setPrintingFunction (printingFn);
    activationFns [1][0][0]->setPrintingFunction (printingFn);
    activationFns [0][0][0]->setMutationFunction (mutationFn);
    activationFns [0][0][1]->setMutationFunction (mutationFn);
    activationFns [1][1][0]->setMutationFunction (mutationFn);
    activationFns [1][1][1]->setMutationFunction (mutationFn);
    activationFns [0][1][0]->setMutationFunction (mutationFn);
    activationFns [1][0][0]->setMutationFunction (mutationFn);

    unsigned int N_ConnInit = 40;
    double probRecuInit = 0.0;
    double weightExtremumInit = 20.0;
    unsigned int maxRecuInit = 0;
    double speciationThreshInit = 20.0;
    distanceFn dstType = CONVENTIONAL;
    unsigned int threshGensSinceImproved = 15;
    return pneatm::Population<myInt, myFloat> (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger, dstType, speciationThreshInit, threshGensSinceImproved, "stats.csv");
}

pneatm::Population<myInt, myFloat> LoadPopulation (const std::string& filename, spdlog::logger* logger, const std::string& stats_filename) {
    // bias values
    std::vector<void*> bias_init;
    myFloat* unitValueFLOAT = new myFloat (1.0f);
    myInt* unitValueINT = new myInt (1);
    bias_init.push_back ((void*) unitValueINT);
    bias_init.push_back ((void*) unitValueFLOAT);

    // reset values
    std::vector<void*> resetValues;
    myFloat* nullValueFLOAT = new myFloat (0.0f);
    myInt* nullValueINT = new myInt (0);
    resetValues.push_back ((void*) nullValueINT);
    resetValues.push_back ((void*) nullValueFLOAT);

    // activation functions setup
    std::vector<std::vector<std::vector<ActivationFnBase*>>> activationFns;
    activationFns.push_back ({});
    activationFns.push_back ({});
    activationFns [0].push_back ({});
    activationFns [0].push_back ({});
    activationFns [1].push_back ({});
    activationFns [1].push_back ({});
    activationFns [0][0].push_back (new ActivationFn<myInt, myInt> ());
    activationFns [0][0].push_back (new ActivationFn<myInt, myInt> ());
    activationFns [1][1].push_back (new ActivationFn<myFloat, myFloat> ());
    activationFns [1][1].push_back (new ActivationFn<myFloat, myFloat> ());
    activationFns [0][1].push_back (new ActivationFn<myInt, myFloat> ());
    activationFns [1][0].push_back (new ActivationFn<myFloat, myInt> ());
    activationFns [0][0][0]->setFunction ((void*) &identity_int2int);   // identity function MUST BE the first
    activationFns [0][0][1]->setFunction ((void*) &sigmoid_int2int);
    activationFns [1][1][0]->setFunction ((void*) &identity_float2float);   // identity function MUST BE the first
    activationFns [1][1][1]->setFunction ((void*) &sigmoid_float2float);
    activationFns [0][1][0]->setFunction ((void*) &sigmoid_int2float);
    activationFns [1][0][0]->setFunction ((void*) &sigmoid_float2int);
    activationFns [0][0][0]->setPrintingFunction (printingFn);  // for all the activation functions, the printing and mutation functions are the sames
    activationFns [0][0][1]->setPrintingFunction (printingFn);
    activationFns [1][1][0]->setPrintingFunction (printingFn);
    activationFns [1][1][1]->setPrintingFunction (printingFn);
    activationFns [0][1][0]->setPrintingFunction (printingFn);
    activationFns [1][0][0]->setPrintingFunction (printingFn);
    activationFns [0][0][0]->setMutationFunction (mutationFn);
    activationFns [0][0][1]->setMutationFunction (mutationFn);
    activationFns [1][1][0]->setMutationFunction (mutationFn);
    activationFns [1][1][1]->setMutationFunction (mutationFn);
    activationFns [0][1][0]->setMutationFunction (mutationFn);
    activationFns [1][0][0]->setMutationFunction (mutationFn);

    return pneatm::Population<myInt, myFloat> (filename, bias_init, resetValues, activationFns, logger, stats_filename);
}

std::function<pneatm::mutationParams_t (double)> SetupMutationParametersMaps () {
    pneatm::mutationParams_t explorationSet;
    explorationSet.nodes.rate = 0.06;
    explorationSet.nodes.monotypedRate = 0.5;
    explorationSet.nodes.monotyped.maxIterationsFindConnection = 100;
    explorationSet.nodes.bityped.maxRecurrencyEntryConnection = 0;
    explorationSet.nodes.bityped.maxIterationsFindNode = 100;
    explorationSet.activation_functions.rate = 0.09;
    explorationSet.connections.rate = 0.06;
    explorationSet.connections.reactivateRate = 0.6;
    explorationSet.connections.maxRecurrency = 0;
    explorationSet.connections.maxIterations = 100;
    explorationSet.connections.maxIterationsFindNode = 100;
    explorationSet.weights.rate = 0.06;
    explorationSet.weights.fullChangeRate = 0.4;
    explorationSet.weights.perturbationFactor = 0.2;
    pneatm::mutationParams_t refinementSet;
    refinementSet.nodes.rate = 0.05;
    refinementSet.nodes.monotypedRate = 0.5;
    refinementSet.nodes.monotyped.maxIterationsFindConnection = 100;
    refinementSet.nodes.bityped.maxRecurrencyEntryConnection = 0;
    refinementSet.nodes.bityped.maxIterationsFindNode = 100;
    refinementSet.activation_functions.rate = 0.08;
    refinementSet.connections.rate = 0.05;
    refinementSet.connections.reactivateRate = 0.6;
    refinementSet.connections.maxRecurrency = 0;
    refinementSet.connections.maxIterations = 100;
    refinementSet.connections.maxIterationsFindNode = 100;
    refinementSet.weights.rate = 0.05;
    refinementSet.weights.fullChangeRate = 0.3;
    refinementSet.weights.perturbationFactor = 0.2;
    return [=] (double fitness) -> pneatm::mutationParams_t {
        // Here, the mutation map is very basic: if the genome is pretty good, we just refine his network, else we explore new networks
        if (fitness > 400.0) {
            return refinementSet;
        }
        return explorationSet;
    };
}

#endif  // SETUP_HPP