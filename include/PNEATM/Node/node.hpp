#ifndef NODE_HPP
#define NODE_HPP

#include <PNEATM/Node/node_base.hpp>
#include <functional>
#include <iostream>
#include <cstring>
#include <memory>

/* HEADER */

namespace pneatm {

/**
 * @brief A template class representing a node in a neural network.
 *
 * The `Node` class is a template class representing a node in a neural network.
 * It is derived from the `NodeBase` abstract base class and provides implementations
 * for the virtual functions defined in the base class.
 *
 * @tparam T_in The input data type for the node.
 * @tparam T_out The output data type for the node.
 */
template <typename T_in, typename T_out>
class Node : public NodeBase {
	public:
		/**
		 * @brief Constructor for the Node class.
		 * @param is_monotyped A boolean indicating if the node is monotyped. (default is false)
		 */
		Node (bool is_monotyped = false);

		/**
		 * @brief Destructor for the Node class.
		 */
		~Node () {};

		/**
		 * @brief Set the activation function for the node.
		 * @param f A pointer to the activation function to be set.
		 */
		void setActivationFn (void* f) override;

		/**
		 * @brief Set the reset value for the node.
		 * @param value A pointer to the reset value to be set.
		 */
		void setResetValue (void* value) override;

		/**
		 * @brief Load an input value to the node.
		 * @param value A pointer to the input value to be loaded.
		 */
		void loadInput (void* value) override;

		/**
		 * @brief Add a value to the node's input with a scalar factor.
		 * @param value A pointer to the value to be added to the input.
		 * @param scalar The scalar factor to multiply the input value with.
		 */
		void AddToInput (void* value, double scalar) override;	// TODO: too dirty

		/**
		 * @brief Get the output value of the node.
		 * @return A pointer to the output value of the node.
		 */
		void* getOutput () override;

		/**
		 * @brief Process the node to compute its output value.
		 */
		void process () override;

		/**
		 * @brief Reset the node to its initial state.
		 */
		void reset () override;

		/**
		 * @brief Create a clone of the node.
		 * @return A unique pointer to the cloned node.
		 */
		std::unique_ptr<NodeBase> clone () override;

		/**
		 * @brief Print information about the node.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (std::string prefix = "") override;

	private:
		T_in input;
		T_out output;
		std::function<T_out (T_in)> func;

		T_in resetValue;
};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename T_in, typename T_out>
Node<T_in, T_out>::Node (bool is_monotyped) {
	if (is_monotyped) {
		func = [] (T_in input) -> T_out {	// default activation function is the identity (useful for bias/inputs/outputs)
			return input;
		};
		index_activation_fn = 0;	// default activation function (identity) id is 0
	}
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setActivationFn (void* f) {
	func = *static_cast<std::function<T_out (T_in)>*> (f);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setResetValue (void* value) {
	resetValue = *static_cast<T_in*> (value);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::loadInput (void* value) {
	input = *static_cast<T_in*> (value);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::AddToInput (void* value, double scalar) {
	input += *static_cast<T_in*> (value) * scalar;
}

template <typename T_in, typename T_out>
void* Node<T_in, T_out>::getOutput () {
	return static_cast<void*> (&output); 
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::process () {
	output = func (input);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::reset () {
	input = resetValue;
}

template <typename T_in, typename T_out>
std::unique_ptr<NodeBase> Node<T_in, T_out>::clone () {
	std::unique_ptr<NodeBase> node =  std::make_unique<Node<T_in, T_out>> ();

	node->id = id;
	node->innovId = innovId;
	node->layer = layer;
	node->index_T_in = index_T_in;
	node->index_T_out = index_T_out;
	node->setResetValue (static_cast<void*> (&resetValue));
	node->setActivationFn (static_cast<void*> (&func));
	node->loadInput (static_cast<void*> (&input));
	node->process ();

	return node;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::print (std::string prefix) {
	std::cout << prefix << "ID: " << id << std::endl;
	std::cout << prefix << "Innovation ID: " << innovId << std::endl;
	std::cout << prefix << "Layer: " << layer << std::endl;
	std::cout << prefix << "Input Type ID: " << index_T_in << std::endl;
	std::cout << prefix << "Output Type ID: " << index_T_out << std::endl;
	std::cout << prefix << "Activation Function ID: " << index_activation_fn << std::endl;
	std::cout << prefix << "Current Input Value: " << input << std::endl;
	std::cout << prefix << "Current Output Value: " << output << std::endl;
	std::cout << prefix << "Reset Value: " << resetValue << std::endl;
}

#endif	// NODE_HPP