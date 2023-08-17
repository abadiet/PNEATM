#ifndef NODE_HPP
#define NODE_HPP

#include <PNEATM/Node/node_base.hpp>
#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
#include <PNEATM/Node/Activation_Function/activation_function.hpp>
#include <PNEATM/utils.hpp>
#include <functional>
#include <iostream>
#include <cstring>
#include <memory>
#include <fstream>

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
		 */
		Node ();

		/**
		 * @brief Destructor for the Node class.
		 */
		~Node () {};

		/**
		 * @brief Set the activation function for the node.
		 * @param actfn A pointer to the activation function to be set.
		 */
		void setActivationFn (std::unique_ptr<ActivationFnBase> actfn) override;

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
		 * @brief Get the output value of the node at a specific time.
		 * @param depth The output's depth (e.g 0 stands for the current output and 3 means 3 calls ro runNetwork later). (default is 0)
		 * @return A pointer to the output value of the node at the given time..
		 */
		void* getOutput (unsigned int depth = 0) override;

		/**
		 * @brief Process the node to compute its output value.
		 */
		void process () override;

		/**
		 * @brief Mutate the activation function's parameters.
		 * @param fitness The current genome's fitness
		 */
		void mutate (double fitness) override;

		/**
		 * @brief Reset the node to its initial state.
		 * @param resetMemory `true` to reset memory too, `false` else. (default is `true`)
		 */
		void reset (bool resetMemory = true) override;

		/**
		 * @brief Create a clone of the node.
		 * @return A unique pointer to the cloned node.
		 */
		std::unique_ptr<NodeBase> clone () override;

		/**
		 * @brief Print information about the node.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (const std::string& prefix = "") const override;

		/**
		 * @brief Serialize the Node instance to an output file stream.
		 * @param outFile The output file stream to which the Node instance will be written.
		 */
		void serialize (std::ofstream& outFile) const override;

		/**
		 * @brief Deserialize a Node instance from an input file stream.
		 * @param inFile The input file stream from which the Node instance will be read.
		 * @param activationFns The activation functions (e.g., activationFns[i][j] is a pointer to an activation function that takes an input of type of index i and return a type of index j output).
		 */
		void deserialize (std::ifstream& inFile, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns) override;

	private:
		T_in input;
		std::vector<T_out> outputs;
		unsigned int N_outputs;
		std::unique_ptr<ActivationFn<T_in, T_out>> activation_fn;

		T_in resetValue;
};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename T_in, typename T_out>
Node<T_in, T_out>::Node () :
	N_outputs (0),
	activation_fn (std::make_unique<ActivationFn<T_in, T_out>> ())
{}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setActivationFn (std::unique_ptr<ActivationFnBase> actfn) {
	activation_fn = std::unique_ptr<ActivationFn<T_in, T_out>> (static_cast<ActivationFn<T_in, T_out>*> (actfn.release ()));
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
void* Node<T_in, T_out>::getOutput (unsigned int depth) {
	for (unsigned int k = (unsigned int) outputs.size (); k <= depth; k++) {
		// the depth is too high for the number of outputs: we create empty outputs to be able to return the pointer to the upcoming output
		outputs.push_back (T_out ());
	}
	return static_cast<void*> (&outputs [(unsigned int) outputs.size () - 1 - depth]);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::process () {
	if ((unsigned int) outputs.size () <= N_outputs) {	// equivalent to == as < is impossible
		outputs.push_back (activation_fn->process (input));
	} else {
		outputs [N_outputs] = activation_fn->process (input);
	}
	N_outputs ++;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::mutate (double fitness) {
	activation_fn->mutate (fitness);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::reset (bool resetMemory) {
	input = resetValue;
	if (resetMemory) {
		N_outputs = 0;
		outputs.clear ();
	}
}

template <typename T_in, typename T_out>
std::unique_ptr<NodeBase> Node<T_in, T_out>::clone () {
	std::unique_ptr<NodeBase> node =  std::make_unique<Node<T_in, T_out>> ();

	node->id = id;
	node->innovId = innovId;
	node->layer = layer;
	node->index_T_in = index_T_in;
	node->index_T_out = index_T_out;
	node->index_activation_fn = index_activation_fn;
	node->setResetValue (static_cast<void*> (&resetValue));
	node->setActivationFn (activation_fn->clone (true));	// note that we keep parameters as they are here
	node->loadInput (static_cast<void*> (&input));

	return node;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::print (const std::string& prefix) const {
	std::cout << prefix << "ID: " << id << std::endl;
	std::cout << prefix << "Innovation ID: " << innovId << std::endl;
	std::cout << prefix << "Layer: " << layer << std::endl;
	std::cout << prefix << "Input Type ID: " << index_T_in << std::endl;
	std::cout << prefix << "Output Type ID: " << index_T_out << std::endl;
	std::cout << prefix << "Activation Function ID: " << index_activation_fn << std::endl;
	std::cout << prefix << "Is Useful in the Network: " << is_useful << std::endl;
	std::cout << prefix << "Current Input Value: " << input << std::endl;
	std::cout << prefix << "Outputs Values (younger first): ";
	for (unsigned int i = 0; i < N_outputs; i++) {
		std::cout << outputs [i];
		if (i < N_outputs - 1) {
			std::cout << " ~ ";
		} else {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	std::cout << prefix << "Reset Value: " << resetValue << std::endl;
	std::cout << prefix << "Activation Function Parameters: ";
	activation_fn->print (prefix);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::serialize (std::ofstream& outFile) const {
	Serialize (id, outFile);
	Serialize (innovId, outFile);
	Serialize (is_useful, outFile);
	Serialize (layer, outFile);
	Serialize (index_T_in, outFile);
	Serialize (index_T_out, outFile);
	Serialize (index_activation_fn, outFile);
	activation_fn->serialize (outFile);
	Serialize (input, outFile);
	Serialize (outputs, outFile);
	Serialize (N_outputs, outFile);
	Serialize (resetValue, outFile);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::deserialize (std::ifstream& inFile, const std::vector<std::vector<std::vector<ActivationFnBase*>>>& activationFns) {
	Deserialize (id, inFile);
	Deserialize (innovId, inFile);
	Deserialize (is_useful, inFile);
	Deserialize (layer, inFile);
	Deserialize (index_T_in, inFile);
	Deserialize (index_T_out, inFile);
	Deserialize (index_activation_fn, inFile);
	setActivationFn (activationFns [index_T_in][index_T_out][index_activation_fn]->clone (true));	// clone with parameters doesn't effect anything has we'll overwrite those parameters later
	activation_fn->deserialize (inFile);	// overwrite parameters
	Deserialize (input, inFile);
	Deserialize (outputs, inFile);
	Deserialize (N_outputs, inFile);
	Deserialize (resetValue, inFile);
}

#endif	// NODE_HPP