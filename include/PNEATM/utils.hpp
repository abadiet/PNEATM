#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <vector>
#include <unordered_map>

#define UNUSED(expr) do { (void) (expr); } while (0)

namespace pneatm {

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
inline bool Eq_Double (double a, double b, double epsi = 1e-12) {
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
inline double Random_Double (double a, double b, bool a_included = true, bool b_included = true) {
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
inline unsigned int Random_UInt (unsigned int a, unsigned int b) {
    return rand () % (b - a + 1) + a;
}

/**
 * @brief Serialize a single object of type T to an output file stream.
 * @tparam T The type of the object to be serialized.
 * @param var The object to be serialized.
 * @param outFile The output file stream to which the object will be written.
 */
template <typename T>
void Serialize (const T& var, std::ofstream& outFile) {
    outFile.write (reinterpret_cast<const char*> (&var), sizeof (var));
}

/**
 * @brief Serialize a vector of type T to an output file stream.
 * @tparam T The type of the vector elements to be serialized.
 * @param var The vector to be serialized.
 * @param outFile The output file stream to which the vector will be written.
 */
template <typename T>
void Serialize (const std::vector<T>& var, std::ofstream& outFile) {
    size_t size = var.size ();
    outFile.write (reinterpret_cast<const char*> (&size), sizeof (size));
    for (const auto& elem : var) {
        Serialize (elem, outFile);
    }
}

/**
 * @brief Serialize an unordered_map of type T with key of type unsigned int to an output file stream.
 * @tparam T The type of the vector elements to be serialized.
 * @param var The map to be serialized.
 * @param outFile The output file stream to which the map will be written.
 */
template <typename T>
void Serialize (const std::unordered_map<unsigned int, T>& var, std::ofstream& outFile) {
    size_t size = var.size ();
    outFile.write (reinterpret_cast<const char*> (&size), sizeof (size));
    for (const std::pair<const unsigned int, T>& elem : var) {
        Serialize (elem.first, outFile);
        Serialize (elem.second, outFile);
    }
}

/**
 * @brief Deserialize a single object of type T from an input file stream.
 * @tparam T The type of the object to be deserialized.
 * @param var The object to hold the deserialized data.
 * @param inFile The input file stream from which the object will be read.
 */
template <typename T>
void Deserialize (T& var, std::ifstream& inFile) {
    inFile.read(reinterpret_cast<char*> (&var), sizeof (var));
}

/**
 * @brief Deserialize a vector of type T from an input file stream.
 * @tparam T The type of the vector elements to be deserialized.
 * @param var The vector to hold the deserialized data.
 * @param inFile The input file stream from which the vector will be read.
 */
template <typename T>
void Deserialize (std::vector<T>& var, std::ifstream& inFile) {
    size_t size;
    inFile.read (reinterpret_cast<char*> (&size), sizeof (size));
    var.clear ();
    var.reserve (size);
    for (size_t i = 0; i < size; i++) {
        Deserialize (var [i], inFile);
    }
}

/**
 * @brief Deserialize an unordered_map of type T with key of type unsigned int from an input file stream.
 * @tparam T The type of the vector elements to be deserialized.
 * @param var The map to hold the deserialized data.
 * @param inFile The input file stream from which the map will be read.
 */
template <typename T>
void Deserialize (std::unordered_map<unsigned int, T>& var, std::ifstream& inFile) {
    size_t size;
    inFile.read (reinterpret_cast<char*> (&size), sizeof (size));
    var.clear ();
    var.reserve (size);
    for (size_t i = 0; i < size; i++) {
        unsigned int key;
        T value;
        Deserialize (key, inFile);
        Deserialize (value, inFile);
        var.insert (std::make_pair (key, value));
    }
}


}

#endif	// UTILS_HPP