#include <snake.hpp>

Snake::Snake (unsigned int playgroundSize) :
    playgroundSize (playgroundSize)
{
    reset ();
}

void Snake::reset () {
    score = 0;
    curMvmt = rand() % 4;
    snake.push_back (
        {
            rand() % (playgroundSize),
            rand() % (playgroundSize)
        }
    );
    fruit.x = rand() % (playgroundSize);
    fruit.y = rand() % (playgroundSize);
    while (isPosBusy (fruit.x, fruit.y, false)) {
        fruit.x = rand() % (playgroundSize);
        fruit.y = rand() % (playgroundSize);
    }
}

std::vector<float> Snake::getAIInputs () {
    std::vector<float> AIinputs;
    // playground bounds
    AIinputs.push_back ((float) snakeEyes((0 + 2 * curMvmt) % 8, false));
    AIinputs.push_back ((float) snakeEyes((1 + 2 * curMvmt) % 8, false));
    AIinputs.push_back ((float) snakeEyes((2 + 2 * curMvmt) % 8, false));
    AIinputs.push_back ((float) snakeEyes((3 + 2 * curMvmt) % 8, false));
    AIinputs.push_back ((float) snakeEyes((4 + 2 * curMvmt) % 8, false));
    AIinputs.push_back ((float) snakeEyes((5 + 2 * curMvmt) % 8, false));
    AIinputs.push_back ((float) snakeEyes((6 + 2 * curMvmt) % 8, false));
    
    // fruit
    AIinputs.push_back ((float) snakeEyes((0 + 2 * curMvmt) % 8, true));
    AIinputs.push_back ((float) snakeEyes((1 + 2 * curMvmt) % 8, true));
    AIinputs.push_back ((float) snakeEyes((2 + 2 * curMvmt) % 8, true));
    AIinputs.push_back ((float) snakeEyes((3 + 2 * curMvmt) % 8, true));
    AIinputs.push_back ((float) snakeEyes((4 + 2 * curMvmt) % 8, true));
    AIinputs.push_back ((float) snakeEyes((5 + 2 * curMvmt) % 8, true));
    AIinputs.push_back ((float) snakeEyes((6 + 2 * curMvmt) % 8, true));

    return AIinputs;
}

bool Snake::run (std::vector<float> inputs) {
    /* The main function of the game: move the snake's part relatively to the inputs, refresh state, add fruits ... */
    // new Mvmt
    if (inputs[2] > 0.5 && inputs[2] > inputs[1] && inputs[2] > inputs[0]) {   // turn right
        curMvmt = (curMvmt + 1) % 4;
    } else {
        if (inputs[1] > 0.5 && inputs[1] > inputs[0] && inputs[1] > inputs[2]) {   // turn left
            curMvmt = (curMvmt - 1 + 4) % 4;  // + 4 to avoid having negative curMvmt
        } else {
            // consider inputs[0] dominent, is that a good idea ? idk
        }
    }

    // move snake
    vecXY_t newDot;
    newDot.x = snake.back ().x;
    newDot.y = snake.back ().y;
    switch (curMvmt) {
        case 0:
            newDot.x += 1;
        break;
        case 1:
            newDot.y += 1;
        break;
        case 2:
            newDot.x -= 1;
        break;
        case 3:
            newDot.y -= 1;
        break;
        default :
            std::cout << "Error: invalid curMvmt: " << curMvmt << std::endl;
            throw 0;
        break;
    }

    snake.erase (snake.begin() + 0);

    // new snake's state
    if (isPosBusy (newDot.x, newDot.y, true)) {
        if (isPosBusy(newDot.x, newDot.y, false)) {
            return true;   // game finsished, return fitness
        } else {    // the snake eat the fruit
            score += 100.0f;
            if ((unsigned int) snake.size () == playgroundSize * playgroundSize - 1) {   // if the snake has the maximum lenght possible
                return true;   // return fitness
            }
            // choose fruit position on an empty place
            fruit.x = rand() % (playgroundSize);
            fruit.y = rand() % (playgroundSize);
            while (isPosBusy (newDot.x, newDot.y, false)) {        
                fruit.x = rand() % (playgroundSize);
                fruit.y = rand() % (playgroundSize);
            }
            snake.insert(snake.begin() + 0, snake[0]);   // double the last dot too avoid to remove it the next frame
        }
    }

    snake.push_back(newDot);

    score += 0.1f;

    return false;  // game not finished
}

unsigned int Snake::snakeEyes(unsigned int orientation, bool detectFruit) {
    /* Return the number of empty case in the orientation relatively to playground (not movement!) */
    const unsigned int curXHeadPos = snake.back ().x;
    const unsigned int curYHeadPos = snake.back ().y;
    int Xstep, Ystep;
    switch (orientation) {
        case 0: // ↖
            Xstep = -1;
            Ystep = 1;
        break;
        case 1: // ↑
            Xstep = 0;
            Ystep = 1;
        break;
        case 2: // ↗
            Xstep = 1;
            Ystep = 1;
        break;
        case 3: // →
            Xstep = 1;
            Ystep = 0;
        break;
        case 4: // ↘
            Xstep = 1;
            Ystep = -1;
        break;
        case 5: // ↓
            Xstep = 0;
            Ystep = -1;
        break;
        case 6: // ↙
            Xstep = -1;
            Ystep = -1;
        break;
        case 7: // ←
            Xstep = -1;
            Ystep = 0;
        break;
        default :
            std::cout << "Error: invalid orientation: " << orientation << std::endl;
            throw 0;
        break;
    }
    unsigned int i = 1;
    while (!isPosBusy (curXHeadPos + i * Xstep, curYHeadPos + i * Ystep, detectFruit)) {
        i ++;
    }
    i --;
    return i;
}

bool Snake::isPosBusy (int x, int y, bool detectFruit) {
    /* Check if something is at this position */

    // there is a wall
    if (x < 0 || x > (int) playgroundSize - 1 || y < 0 || y > (int) playgroundSize - 1) {
        return true;
    }

    // there is the snake's body
    for (size_t i = 0; i < snake.size (); i++) {
        if (snake [i].x == (unsigned int) x && snake [i].y == (unsigned int) y) {
            return true;
        }
    }

    // there is the fruit
    return detectFruit && fruit.x == (unsigned int) x && fruit.y == (unsigned int) y;
}

void Snake::drawPlaygroundConsole () {
    /* Print in the console the playground with circle for snake's part and sharp for fruits */
    for (unsigned int i = 0; i < playgroundSize; i++) {
        for (unsigned int j = 0; j < playgroundSize; j++) {
            if (isPosBusy (i, j, true)) {
                if (isPosBusy (i, j, false)) {
                    std::cout << " o ";
                } else {
                    std::cout << " # ";
                }
            } else {
                std::cout << " . ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Snake::drawPlaygroundSFML (sf::RenderWindow* window, float timeUpsSeconds) {
    sf::Clock clock;
    sf::Time accumulator = sf::Time::Zero;
    sf::Time ups = sf::seconds(timeUpsSeconds);

    while (window->isOpen() && accumulator < ups) {
        sf::Event event;
        while (window->pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                window->close();
            }
        }

        window->clear(sf::Color::Black);
        std::vector<std::vector<sf::CircleShape>> dots;

        float displayStep;
        float displayFirstY;
        float displayFirstX;
        if (window->getSize().x > window->getSize().y) {
            displayStep = 0.8f * (float) (window->getSize().y / playgroundSize);
            displayFirstY = 0.1f * (float) window->getSize().y;
            displayFirstX = (float) (window->getSize().x - 0.8 * window->getSize().y) / 2.0f;
        } else {
            displayStep = 0.8f * (float) window->getSize().x / (float) playgroundSize;
            displayFirstX = 0.1f * (float) window->getSize().x;
            displayFirstY = (float) (window->getSize().y - 0.8 * window->getSize().x) / 2.0f;
        }
        float snakeDotsRadius = 0.8f * (displayStep / 2.0f);
        float fruitDotsRadius = 0.8f * (displayStep / 2.0f);
        float emptyDotsRadius = 0.1f * (displayStep / 2.0f);
        
        for (unsigned int i = 0; i < playgroundSize; i++) {
            dots.push_back ({});
            for (unsigned int j = 0; j < playgroundSize; j++) {
                dots [i].push_back (sf::CircleShape ());
                if (isPosBusy(i, j, true)) {
                    if (isPosBusy(i, j, false)) {
	                    dots[i][j].setRadius(snakeDotsRadius);
	                    dots[i][j].setPosition({displayFirstX + displayStep * (float) j - snakeDotsRadius, displayFirstY + displayStep * (float) i - snakeDotsRadius});
	                    dots[i][j].setFillColor(sf::Color::Green);
                    } else {
	                    dots[i][j].setRadius(fruitDotsRadius);
	                    dots[i][j].setPosition({displayFirstX + displayStep * (float) j - fruitDotsRadius, displayFirstY + displayStep * (float) i - fruitDotsRadius});
	                    dots[i][j].setFillColor(sf::Color::Red);
                    }
                } else {
                    dots[i][j].setRadius(emptyDotsRadius);
                    dots[i][j].setPosition({displayFirstX + displayStep * (float) j - emptyDotsRadius, displayFirstY + displayStep * (float) i - emptyDotsRadius});
                    dots[i][j].setFillColor(sf::Color::White);
                }
            }
        }

        for (unsigned int i = 0; i < playgroundSize; i++) {
            for (unsigned int j = 0; j < playgroundSize; j++) {
	            window->draw(dots[i][j]);
            }
        }

        window->display();

        accumulator += clock.restart();
    }
}