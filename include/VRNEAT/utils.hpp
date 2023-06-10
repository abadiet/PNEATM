#ifndef UTILS_HPP
#define UTILS_HPP

#include <VRNEAT/Node/node_base.hpp>
#include <VRNEAT/Node/node.hpp>
#include <memory>

namespace vrneat {

template <typename T1, typename T2, typename... Args>
struct CreateNode;

bool Eq_Float (float a, float b, float epsi = 1e-6f);
float Random_Float (float a, float b, bool a_included = true, bool b_included = true);

}

#endif	// UTILS_HPP