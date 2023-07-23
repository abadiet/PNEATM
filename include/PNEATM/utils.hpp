#ifndef UTILS_HPP
#define UTILS_HPP

#include <PNEATM/Node/node_base.hpp>
#include <PNEATM/Node/node.hpp>
#include <memory>
#include <cstdlib>

#define UNUSED(expr) do { (void) (expr); } while (0)

namespace pneatm {

/**
 * @brief A struct for creating nodes based on input and output types.
 *
 * The `CreateNode` struct provides static template member functions to create nodes based on
 * the input and output types. It uses variadic templates to handle multiple input and output types.
 */
struct CreateNode {
    /**
     * @brief Create a node with specified input and output types..
     * @tparam T1 The first type.
     * @tparam T2 The second type.
     * @tparam Args Variadic template arguments that represent the remaining types.
     * @param iT_in The index of the input type (used for recursive template calls).
     * @param iT_out The index of the output type (used for recursive template calls).
     * @param mono_type Set to true for mono-type nodes, false for bi-type nodes. (default is true)
     * @param T2_is_first Set to true if T2 is the input type. (default is false)
     * @return A unique pointer to the created NodeBase instance.
     */
    template <typename T1, typename T2, typename... Args>
    static std::unique_ptr<NodeBase> get (size_t iT_in, size_t iT_out, bool mono_type = true, bool T2_is_first = false) {
        if (iT_in == 0 && iT_out == 0) {
            if (mono_type) {
                return std::make_unique<Node <T1, T1>> (iT_in == iT_out);
            }
            if (T2_is_first) {
                return std::make_unique<Node <T2, T1>> (iT_in == iT_out);
            } else {
                return std::make_unique<Node <T1, T2>> (iT_in == iT_out);
            }
        }
        size_t new_iT_in = iT_in;
        size_t new_iT_out = iT_out;
        if (new_iT_in > 0) {
            new_iT_in --;
            if (new_iT_out > 0) {
                new_iT_out --;
                // both T_in and T_out are not found
                // Note that if T2 is the last type (aka Args is nothing), the next line will call template <typename T> static CreateNode::NodeBase* get(size_t iT_in, size_t iT_out) which is what we expect as T_in = T_out
                return CreateNode::get<T2, Args...>(new_iT_in, new_iT_out);
            } else {
                // T_out is found and is T1, we keep it
                // It is not a mono type node
                return CreateNode::get<T1, T2, Args...>(new_iT_in, new_iT_out, false, true);
            }
        } else {
            new_iT_out --;
            // T_out is not found, because we would return something if both were found
            // However, T_in is found and is T1, we keep it in first place
            // It is not a mono type node
            return CreateNode::get<T1, T2, Args...>(new_iT_in, new_iT_out, false, false);
        }
    }

    /**
     * @brief Create a node with a single input/output type.
     * @tparam T The input and output type.
     * @param iT_in The index of the input type.
     * @param iT_out The index of the output type.
     * @return A unique pointer to the created NodeBase instance.
     */
    template <typename T>
    static std::unique_ptr<NodeBase> get (size_t iT_in, size_t iT_out) {
        return std::make_unique<Node <T, T>> (iT_in == iT_out);
    }
};

/**
 * @brief Check if two double values are equal within a given epsilon tolerance.
 *
 * This function compares two double values, `a` and `b`, to determine if they are equal within the
 * specified epsilon tolerance `epsi`. It is common to compare double values using an epsilon to
 * account for floating-point precision errors. The function returns true if the absolute difference
 * between `a` and `b` is less than or equal to `epsi`, indicating equality within the tolerance.
 *
 * @param a The first double value to compare.
 * @param b The second double value to compare.
 * @param epsi The epsilon tolerance for the comparison. (default is 1e-12)
 * @return True if `a` is approximately equal to `b` within the epsilon tolerance, otherwise false.
 */
bool Eq_Double (double a, double b, double epsi = 1e-12) {
    return a < b + epsi && a > b - epsi;
}

/**
 * @brief Generate a random double value within the specified range.
 *
 * This function generates a random double value within the range [a, b]. The function uses the standard
 * C library's `rand()` function to generate a random floating-point value and scales it to a value within the range.
 * By default, both endpoints `a` and `b` are included in the range, but this can be controlled using
 * the `a_included` and `b_included` parameters.
 *
 * @param a The lower bound of the range.
 * @param b The upper bound of the range.
 * @param a_included Set to true to include the lower bound `a` in the range. (default is true)
 * @param b_included Set to true to include the upper bound `b` in the range. (default is true)
 * @return A random double value within the specified range [a, b].
 */
double Random_Double (double a, double b, bool a_included = true, bool b_included = true) {
    return (
            ((double) rand () + (double) !a_included)
        ) / (
            (double) (RAND_MAX) + (double) !a_included + (double) !b_included
        ) * (b - a) + a;
}

/**
 * @brief Generate a random unsigned integer within the specified range.
 *
 * This function generates a random unsigned integer value within the range [a, b]. The function uses
 * the standard C library's `rand()` function to generate a random integer and applies the modulo
 * operation to ensure the generated value falls within the specified range.
 *
 * @param a The lower bound of the range (inclusive).
 * @param b The upper bound of the range (inclusive).
 * @return A random unsigned integer within the specified range [a, b].
 */
unsigned int Random_UInt (unsigned int a, unsigned int b) {
    return rand () % (b - a + 1) + a;
}

}

#endif	// UTILS_HPP