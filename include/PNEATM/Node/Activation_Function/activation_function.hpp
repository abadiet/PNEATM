#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
#include <PNEATM/utils.hpp>
#include <iostream>
#include <cstring>
#include <memory>
#include <fstream>

#define UNUSED(expr) do { (void) (expr); } while (0)


/* HEADER */

// Forward declaration
typedef struct activationFnParams activationFnParams_t;

namespace pneatm {

/**
 * @brief A template class representing an activation function in a neural network.
 *
 * The `ActivationFn` class is a template class representing an activation function in a neural network.
 * It is derived from the `ActivationFnBase` abstract base class and provides implementations
 * for the virtual functions defined in the base class.
 *
 * @tparam T_in The input data type for the activation function.
 * @tparam T_out The output data type for the activation function.
 */
template <typename T_in, typename T_out>
class ActivationFn : public ActivationFnBase {
	public:
		/**
		 * @brief Constructor for the ActivationFn class.
		 *
		 * The constructor initalized the activation function's parameters to their default values and initialized
		 * mutation and printing functions to functions that does nothing.
		 */
		ActivationFn ();

		/**
		 * @brief Destructor for the ActivationFn class.
		 */
		~ActivationFn () {};

		/**
		 * @brief Set the activation function aka the function used to get the node's outpus from its input.
		 * @param func A pointer to the activation function to be set.
		 */
        void setFunction (void* func) override;

		/**
		 * @brief Set the mutation function aka the function used to mutate the activation function's parameters.
		 * @param func A pointer to the mutation function to be set.
		 */
        void setMutationFunction (const std::function<void (activationFnParams_t*, double)>& func) override;

		/**
		 * @brief Set the printing function aka the function used to print the activation function's parameters.
		 * @param func A pointer to the printing function to be set.
		 */
        void setPrintingFunction (const std::function<void (activationFnParams_t*, std::string)>& func) override;

		/**
		 * @brief Create a clone of the class: clone the activation function, the mutation function, the printing function and the parameters (optionally).
		 * @param preserveParameters True if the parameters should be cloned, False else. (default is true)
		 * @return A unique pointer to the cloned node.
		 */
		std::unique_ptr<ActivationFnBase> clone (bool preserveParameters = true) override;

		/**
		 * @brief Process the activation function to compute its output value.
		 */
		T_out process (const T_in& value);

		/**
		 * @brief Mutate the activatoin function's parameters.
		 * @param fitness The current genome's fitness
		 */
		void mutate (double fitness) override;

		/**
		 * @brief Print information about the activation function's parameters.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (const std::string& prefix = "");

        /**
		 * @brief Serialize the AcivationFn instance to an output file stream.
		 * @param outFile The output file stream to which the AcivationFn instance will be written.
		 */
		void serialize (std::ofstream& outFile) override;

		/**
		 * @brief Deserialize a AcivationFn instance from an input file stream.
		 * @param inFile The input file stream from which the AcivationFn instance will be read.
		 */
		void deserialize (std::ifstream& inFile) override;

    private:
		std::function<T_out (T_in, activationFnParams_t*)> processFn;

};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename T_in, typename T_out>
ActivationFn<T_in, T_out>::ActivationFn () {
	mutationFn = [=] (activationFnParams_t* params, double fitness) {
		// default mutation function do nothing
		UNUSED (params);
		UNUSED (fitness);
	};

	printingFn = [=] (activationFnParams_t* params, std::string prefix) {
		// default printing function do nothing
		UNUSED (params);
		UNUSED (prefix);
	};

	params = std::make_unique<activationFnParams_t> ();
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::setFunction (void* func) {
	processFn = *static_cast<std::function<T_out (T_in, activationFnParams_t*)>*> (func);
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::setMutationFunction (const std::function<void (activationFnParams_t*, double)>& func) {
	mutationFn = func;
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::setPrintingFunction (const std::function<void (activationFnParams_t*, std::string)>& func) {
	printingFn = func;
}

template <typename T_in, typename T_out>
std::unique_ptr<ActivationFnBase> ActivationFn<T_in, T_out>::clone (bool preserveParameters) {
	std::unique_ptr<ActivationFnBase> actfun =  std::make_unique<ActivationFn<T_in, T_out>> ();

	if (preserveParameters) {
		*actfun->params = *params;
	}
	actfun->setFunction (static_cast<void*> (&processFn));
	actfun->mutationFn = mutationFn;
	actfun->printingFn = printingFn;

	return actfun;
}

template <typename T_in, typename T_out>
T_out ActivationFn<T_in, T_out>::process (const T_in& value) {
	return processFn (value, params.get ());
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::mutate (double fitness) {
	mutationFn (params.get (), fitness);
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::print (const std::string& prefix) {
	printingFn (params.get (), prefix);
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::serialize (std::ofstream& outFile) {
	Serialize (*params, outFile);
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::deserialize (std::ifstream& inFile) {
	activationFnParams_t params_tmp;
	Deserialize (params_tmp, inFile);
	*params = params_tmp;
}

#endif	// ACTIVATION_FUNCTION_HPP