#include <LRNEAT/utils.hpp>

using namespace neat;

bool Eq_Float (float a, float b, float epsi = 1e-6) {
    return a < b + epsi && a > b - epsi;
}

float Random_Float (float a, float b, bool a_included = true, bool b_included = true) {
    return (
            ((float) rand() + (float) !a_included)
        ) / (
            (float) (RAND_MAX) + (float) !a_included + (float) !b_included
        ) * (b - a) + a;
}