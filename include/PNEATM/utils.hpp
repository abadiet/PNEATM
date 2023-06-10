#ifndef UTILS_HPP
#define UTILS_HPP

#include <PNEATM/Node/node_base.hpp>
#include <PNEATM/Node/node.hpp>
#include <memory>

namespace pneatm {

template <typename T1, typename T2, typename... Args>
struct CreateNode;

bool Eq_Float (float a, float b, float epsi = 1e-6f);
float Random_Float (float a, float b, bool a_included = true, bool b_included = true);

}

#endif	// UTILS_HPP