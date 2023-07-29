#ifndef ACTIVATION_FUNCTION_BASE_HPP
#define ACTIVATION_FUNCTION_BASE_HPP

#include <iostream>
#include <cstring>
#include <memory>

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
        virtual ~ActivationFnBase () {};

        virtual void setFunction (void* func) = 0;

        virtual void setMutationFunction (std::function<void (activationFnParams_t*, double)> func) = 0;

        virtual void setPrintingFunction (std::function<void (activationFnParams_t*, std::string)> func) = 0;

		virtual std::unique_ptr<ActivationFnBase> clone (bool preserveParameters = true) = 0;

		virtual void mutate (double fitness) = 0;

		virtual void print (std::string prefix = "") = 0;

    protected:
        std::unique_ptr<activationFnParams_t> params;
        std::function<void (activationFnParams_t*, double)> mutationFn;
        std::function<void (activationFnParams_t*, std::string)> printingFn;

    template <typename T_in, typename T_out>
	friend class ActivationFn;

};

}

#endif	// ACTIVATION_FUNCTION_BASE_HPP