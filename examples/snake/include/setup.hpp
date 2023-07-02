#ifndef SETUP_HPP
#define SETUP_HPP

#include <iostream>
#include <vector>
#include <snake.hpp>
#include <PNEATM/population.hpp>
#include <PNEATM/genome.hpp>
#include <functional>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
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

#endif  // SETUP_HPP