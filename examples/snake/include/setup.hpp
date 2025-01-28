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
    sf::RenderWindow window(sf::VideoMode(windowSize.x, windowSize.y), "PNEATM - https://github.com/abadiet");

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
std::function<myInt (myInt, activationFnParams_t*)> inputs_int_activation_fn = [] (myInt x, activationFnParams_t* params) -> myInt {
    // identity function
    return x;
    UNUSED (params);
};
std::function<myFloat (myFloat, activationFnParams_t*)> inputs_float_activation_fn = [] (myFloat x, activationFnParams_t* params) -> myFloat {
    // identity function
    return x;
    UNUSED (params);
};
std::function<myFloat (myFloat, activationFnParams_t*)> outputs_float_activation_fn = [] (myFloat x, activationFnParams_t* params) -> myFloat {
    // sigmoid function
    return myFloat ((float) (1.0 / (1.0 + exp(-1.0 * (double) x))));
    UNUSED (params);
};
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
std::function<void (activationFnParams_t*, std::string)> noPrintingFn = [] (activationFnParams_t* params, std::string prefix) -> void {
    UNUSED (params);
    UNUSED (prefix);
};
std::function<void (activationFnParams_t*, std::string)> printingFn = [] (activationFnParams_t* params, std::string prefix) -> void {
    std::cout << prefix << "alpha = " << params->alpha << "   beta = " << params->beta << std::endl;
};

// the mutation function
std::function<void (activationFnParams_t*, double)> mutationFn = [] (activationFnParams_t* params, double fitness) -> void {
    // Here the mutation function reset or perturb the parameters, based on the fitness.
    // The function used determine wether to perturb or reset the parameters is a progressive decreasing function from 80% chance when fitness = 0
    // to 0% when fitness = +inf while the half, 40% chance, is dedicated to fitness = 400.
    const double reset_prob = 0.8 / (1 + (fitness / 400) * (fitness / 400));

    if (Random_Double (0.0, 1.0, true, false) < reset_prob) {
        // reset values
        params->alpha = Random_Double (-10.0, 10.0);
        params->beta = Random_Double (-10.0, 10.0);
    } else {
        // perturb values
        params->alpha += params->alpha * Random_Double (-0.2, 0.2);
        params->beta += params->beta * Random_Double (-0.2, 0.2);
    }
};


/* SETUP MAIN FUNCTIONS */

pneatm::Population<myInt, myFloat>* SetupPopulation (unsigned int popSize, spdlog::logger* logger) {
    // nodes scheme setup
    std::vector<size_t> bias_sch = {1, 1};
    std::vector<size_t> inputs_sch = {14, 1};
    std::vector<size_t> outputs_sch = {0, 1};
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
    // inputs
    std::vector<ActivationFnBase*> inputsActivationFns;
    inputsActivationFns.push_back (new ActivationFn<myInt, myInt> ());   //bias myInt
    inputsActivationFns.back ()->setFunction ((void*) &inputs_int_activation_fn);
    inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    inputsActivationFns.push_back (new ActivationFn<myFloat, myFloat> ());   //bias myFloat
    inputsActivationFns.back ()->setFunction ((void*) &inputs_float_activation_fn);
    inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    for (int k = 0; k < 14; k++) {
        inputsActivationFns.push_back (new ActivationFn<myInt, myInt> ());   // 14 input nodes myInt
        inputsActivationFns.back ()->setFunction ((void*) &inputs_int_activation_fn);
        inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    }
    inputsActivationFns.push_back (new ActivationFn<myFloat, myFloat> ());   // 1 input node myFloat
    inputsActivationFns.back ()->setFunction ((void*) &inputs_float_activation_fn);
    inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    // outputs
    std::vector<ActivationFnBase*> outputsActivationFns;
    outputsActivationFns.push_back (new ActivationFn<myFloat, myFloat> ());   //output myFloat
    outputsActivationFns.back ()->setFunction ((void*) &outputs_float_activation_fn);
    outputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    // hiddens
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
    return new pneatm::Population<myInt, myFloat> (popSize, bias_sch, inputs_sch, outputs_sch, hiddens_sch_init, bias_init, resetValues, activationFns, inputsActivationFns, outputsActivationFns, N_ConnInit, probRecuInit, weightExtremumInit, maxRecuInit, logger);
}

pneatm::Population<myInt, myFloat>* LoadPopulation (const std::string& filename, spdlog::logger* logger, const std::string& stats_filename) {
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
    // inputs
    std::vector<ActivationFnBase*> inputsActivationFns;
    inputsActivationFns.push_back (new ActivationFn<myInt, myInt> ());   //bias myInt
    inputsActivationFns.back ()->setFunction ((void*) &inputs_int_activation_fn);
    inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    inputsActivationFns.push_back (new ActivationFn<myFloat, myFloat> ());   //bias myFloat
    inputsActivationFns.back ()->setFunction ((void*) &inputs_float_activation_fn);
    inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    for (int k = 0; k < 14; k++) {
        inputsActivationFns.push_back (new ActivationFn<myInt, myInt> ());   // 14 input nodes myInt
        inputsActivationFns.back ()->setFunction ((void*) &inputs_int_activation_fn);
        inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    }
    inputsActivationFns.push_back (new ActivationFn<myFloat, myFloat> ());   // 1 input node myFloat
    inputsActivationFns.back ()->setFunction ((void*) &inputs_float_activation_fn);
    inputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    // outputs
    std::vector<ActivationFnBase*> outputsActivationFns;
    outputsActivationFns.push_back (new ActivationFn<myFloat, myFloat> ());   //output myFloat
    outputsActivationFns.back ()->setFunction ((void*) &outputs_float_activation_fn);
    outputsActivationFns.back ()->setPrintingFunction (noPrintingFn);
    // hiddens
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

    return new pneatm::Population<myInt, myFloat> (filename, bias_init, resetValues, activationFns, inputsActivationFns, outputsActivationFns, logger, stats_filename);
}

std::function<pneatm::mutationParams_t (double)> SetupMutationParametersMaps () {
    return [=] (double fitness) -> pneatm::mutationParams_t {
        const double exploration_factor = 2.0 / (1.0 + (fitness / 2000.0) * (fitness / 2000.0)) + 1.0;

        pneatm::mutationParams_t params;
        params.nodes.rate = exploration_factor * 0.05;
        params.nodes.monotypedRate = 0.5;
        params.nodes.monotyped.maxIterationsFindConnection = 100;
        params.nodes.bityped.maxRecurrencyEntryConnection = 0;
        params.nodes.bityped.maxIterationsFindNode = 100;
        params.activation_functions.rate = exploration_factor * 0.08;
        params.connections.rate = exploration_factor * 0.05;
        params.connections.reactivateRate = exploration_factor * 0.6;
        params.connections.maxRecurrency = 0;
        params.connections.maxIterations = 100;
        params.connections.maxIterationsFindNode = 100;
        params.weights.rate = exploration_factor * 0.05;
        params.weights.fullChangeRate = exploration_factor * 0.2;
        params.weights.perturbationFactor = exploration_factor * 0.2;
        return params;
    };
}

#endif  // SETUP_HPP