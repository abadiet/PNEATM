#ifndef UTILS_HPP
#define UTILS_HPP

namespace vrneat {

bool Eq_Float (float a, float b, float epsi = 1e-6f);
float Random_Float (float a, float b, bool a_included = true, bool b_included = true);

}

#endif	// UTILS_HPP