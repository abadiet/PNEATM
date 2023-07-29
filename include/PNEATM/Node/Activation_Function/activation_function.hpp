#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
#include <iostream>
#include <cstring>
#include <memory>

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
		ActivationFn ();

		~ActivationFn () {};

        void setFunction (void* func) override;

        void setMutationFunction (std::function<void (activationFnParams_t*, double)> func) override;

        void setPrintingFunction (std::function<void (activationFnParams_t*, std::string)> func) override;

		std::unique_ptr<ActivationFnBase> clone (bool preserveParameters = true) override;

		T_out process (T_in value);

		void mutate (double fitness) override;

		void print (std::string prefix = "");

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
void ActivationFn<T_in, T_out>::setMutationFunction (std::function<void (activationFnParams_t*, double)> func) {
	mutationFn = func;
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::setPrintingFunction (std::function<void (activationFnParams_t*, std::string)> func) {
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
T_out ActivationFn<T_in, T_out>::process (T_in value) {
	return processFn (value, params.get ());
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::mutate (double fitness) {
	mutationFn (params.get (), fitness);
}

template <typename T_in, typename T_out>
void ActivationFn<T_in, T_out>::print (std::string prefix) {
	printingFn (params.get (), prefix);
}

#endif	// ACTIVATION_FUNCTION_HPP