#ifndef SNAKE_HPP
#define SNAKE_HPP

#include <iostream>
#include <vector>
#include <SFML/Graphics.hpp>
#include <myTypes.hpp>

typedef struct vecXY {
    unsigned int x;
    unsigned int y;
} vecXY_t;

class Snake {
    public:
        Snake (unsigned int playgroundSize);

        void reset ();

        float getScore () {return score;};

        std::vector<myInt> getAIInputs ();
        bool run (myFloat input);

        void drawPlaygroundConsole ();
        void drawPlaygroundSFML (sf::RenderWindow* window, float timeUpsSeconds);

    private:
        float score;
        unsigned int playgroundSize;
        unsigned int curMvmt;
        std::vector<vecXY_t> snake;
        vecXY_t fruit;

        bool isPosBusy (int x, int y, bool detectFruit);
        unsigned int snakeEyes (unsigned int orientation, bool detectFruit);
        
};

#endif  // SNAKE_HPP