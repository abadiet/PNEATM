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

    std::vector<float> AI_Inputs;
    std::vector<float> Snake_Inputs;
    bool isFinished = false;
    unsigned int iteration = 0;
    while (iteration < maxIterationThresh && !isFinished) {
        AI_Inputs = snake.getAIInputs ();

        genome.template loadInputs<float> (AI_Inputs);
        genome.runNetwork ();

        Snake_Inputs = genome.template getOutputs<float> ();

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


/* TYPES */


/* ACTIVATION FUNCTIONS */

std::function<float (float)> sigmoid_float2float = [] (float x) {
    return (1.0f / (1.0f + (float) exp(-1 * 4.09 * x)));
};

#endif  // SETUP_HPP