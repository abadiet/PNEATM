#ifndef UTILS_HPP
#define UTILS_HPP

namespace neat {

bool Eq_Float (float a, float b, float epsi = 1e-10);
float Random_Float (float a, float b, bool a_included = true, bool b_included = true);

}

#endif	// UTILS_HPP