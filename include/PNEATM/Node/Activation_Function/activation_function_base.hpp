#ifndef ACTIVATION_FUNCTION_BASE_HPP
#define ACTIVATION_FUNCTION_BASE_HPP

#include <iostream>
#include <cstring>
#include <memory>
#include <functional>

// Forward declarations
typedef struct activationFnParams activationFnParams_t;

namespace pneatm {

/**
 * @brief Abstract base class representing a generic node's activation function in a neural network.
 *
 * The `ActivationFnBase` class is an abstract base class that defines the interface for a generic activiation function in a neural network.
 * All specific activation function types in the neural network should inherit from this class and implement its virtual functions.
 */
class ActivationFnBase {
    public:
        /**
         * @brief Virtual destructor to ensure proper cleanup in derived classes.
         */
        virtual ~ActivationFnBase () {};

        /**
         * @brief Sets the activation function implementation.
         * @param func A pointer to the activation function to be set.
         */
        virtual void setFunction (void* func) = 0;

        /**
         * @brief Sets the mutation function for the activation function's parameters.
         * @param func The mutation function that modifies the activation function's parameters based on the fitness value.
         */
        virtual void setMutationFunction (std::function<void (activationFnParams_t*, double)> func) = 0;

        /**
         * @brief Sets the printing function for the activation function's parameters.
         * @param func The printing function that displays information about the activation function's parameters.
         */
        virtual void setPrintingFunction (std::function<void (activationFnParams_t*, std::string)> func) = 0;

        /**
         * @brief Creates a clone of the activation function object.
         * @param preserveParameters Set to true if you want to copy the parameters of the current function to the new one, else they are set by the default constructor. (default is true)
         * @return A unique_ptr to the cloned ActivationFnBase object.
         */
		virtual std::unique_ptr<ActivationFnBase> clone (bool preserveParameters = true) = 0;

        /**
         * @brief Mutates the activation function based on the provided fitness value.
         * @param fitness The fitness value.
         */
		virtual void mutate (double fitness) = 0;

        /**
         * @brief Prints information about the activation function.
         * @param prefix A prefix to print before each line. (default is an empty string)
         */
		virtual void print (const std::string& prefix = "") = 0;

        virtual void serialize (std::ofstream& outFile) = 0;

    protected:
        /**
         * @brief The activation function's parameters.
         */
        std::unique_ptr<activationFnParams_t> params;

        /**
         * @brief The parameters' mutation function.
         */
        std::function<void (activationFnParams_t*, double)> mutationFn;

        /**
         * @brief The parameters' printing function.
         */
        std::function<void (activationFnParams_t*, std::string)> printingFn;

    template <typename T_in, typename T_out>
	friend class ActivationFn;

};

}

#endif	// ACTIVATION_FUNCTION_BASE_HPP