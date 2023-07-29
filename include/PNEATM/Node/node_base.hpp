#ifndef NODE_BASE_HPP
#define NODE_BASE_HPP

#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
#include <functional>
#include <iostream>
#include <cstring>
#include <memory>


namespace pneatm {

/**
 * @brief Abstract base class representing a generic node in a neural network.
 *
 * The `NodeBase` class is an abstract base class that defines the interface for a generic node in a neural network.
 * All specific node types in the neural network should inherit from this class and implement its virtual functions.
 */
class NodeBase{
    public:
        virtual ~NodeBase() {};

		/**
		 * @brief Set the activation function for the node.
		 * @param f A pointer to the activation function to be set.
		 */
		virtual void setActivationFn (std::unique_ptr<ActivationFnBase> actfn) = 0;

		/**
		 * @brief Set the reset value for the node.
		 * @param value A pointer to the reset value to be set.
		 */
		virtual void setResetValue (void* value) = 0;

		/**
		 * @brief Load an input value to the node (to use for input and bias nodes only).
		 * @param value A pointer to the input value to be loaded.
		 */
		virtual void loadInput (void* value) = 0;

		/**
		 * @brief Add a value to the node's input with a scalar factor.
		 * @param value A pointer to the value to be added to the input.
		 * @param scalar The scalar factor to multiply the input value with.
		 */
		virtual void AddToInput (void* value, double scalar) = 0;	// TODO: too dirty

		/**
		 * @brief Get the output value of the node.
		 * @return A pointer to the output value of the node.
		 */
		virtual void* getOutput () = 0;

		/**
		 * @brief Process the node to compute its output value.
		 */
		virtual void process () = 0;

		virtual void mutate (double fitness) = 0;

		/**
		 * @brief Reset the node to its initial state.
		 */
		virtual void reset () = 0;

		/**
		 * @brief Create a clone of the node.
		 * @return A unique pointer to the cloned node.
		 */
		virtual std::unique_ptr<NodeBase> clone () = 0;

		/**
		 * @brief Print information about the node.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		virtual void print (std::string prefix = "") = 0;

	protected:
		unsigned int id;
		unsigned int innovId;
		int layer;
		unsigned int index_T_in;
		unsigned int index_T_out;
		unsigned int index_activation_fn;

	template <typename... Args>
	friend class Genome;
	template <typename T_in, typename T_out>
	friend class Node;
};

}

#endif	// NODE_BASE_HPP