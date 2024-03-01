#ifndef NODE_HPP
#define NODE_HPP

#include <PNEATM/Node/node_base.hpp>
#include <PNEATM/Node/Activation_Function/activation_function_base.hpp>
#include <PNEATM/Node/Activation_Function/activation_function.hpp>
#include <PNEATM/circular_buffer.hpp>
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
		 * @param parameters The activation function's parameters. (default is 'nullptr' which does not change the current parameters)
		 */
		void setActivationFn (std::unique_ptr<ActivationFnBase> actfn, activationFnParams_t* parameters = nullptr) override;

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
		 * @brief Setup the vector's outputs.
		 */
		void setupOutputs () override;

		/**
		 * @brief Save the current output t othe saved set.
		 * @param depth The output's depth (e.g 0 stands for the current output and 3 means 3 calls ro runNetwork later). (default is 0)
		 */
		void saveOutput (unsigned int depth = 0) override;

		/**
		 * @brief Get the saved outputs.
		 * @return A pointer to the saved outputs.
		 */
		void* getSavedOutputs () override;

		/**
		 * @brief Process the node to compute its output value.
		 * @return 'false' if the result is NaN, 'true' else.
		 */
		bool process () override;

		/**
		 * @brief Mutate the activation function's parameters.
		 * @param fitness The current genome's fitness
		 */
		void mutate (double fitness) override;

		/**
		 * @brief Reset the node to its initial state.
		 * @param resetMemory `true` to reset memory too, `false` else. (default is `true`)
		 * @param resetBuffer `true` to reset the whole output's buffer too, `false` else. (default is `false`)
		 * @param resetInput `true` to reset the input, `false` else. (default is `true`)
		 */
		void reset (bool resetMemory = true, bool resetBuffer = false, bool resetInput = true) override;

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
		 * @param activationFn A pointer to the activation function.
		 */
		void deserialize (std::ifstream& inFile, ActivationFnBase* activationFn) override;

	private:
		T_in input;
		CircularBuffer<T_out> outputs_buf;
		std::vector<T_out> outputs_saved;
		std::unique_ptr<ActivationFn<T_in, T_out>> activation_fn;
		T_in resetValue;
};

}


/* IMPLEMENTATIONS */

using namespace pneatm;

template <typename T_in, typename T_out>
Node<T_in, T_out>::Node () :
	activation_fn (std::make_unique<ActivationFn<T_in, T_out>> ())
{
	is_useful = false;
	max_depth_recu = 0;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setActivationFn (std::unique_ptr<ActivationFnBase> actfn, activationFnParams_t* parameters) {
	activation_fn = std::unique_ptr<ActivationFn<T_in, T_out>> (static_cast<ActivationFn<T_in, T_out>*> (actfn.release ()));

	if (parameters != nullptr) {
		activation_fn->setParameters (parameters);
	}
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
	return static_cast<void*> (outputs_buf.access_ptr (depth));
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::saveOutput (unsigned int depth) {
	return outputs_saved.push_back (outputs_buf [depth]);
}

template <typename T_in, typename T_out>
void* Node<T_in, T_out>::getSavedOutputs () {
	return static_cast<void*> (&outputs_saved);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setupOutputs () {
	outputs_buf = CircularBuffer<T_out> (max_depth_recu + 1);
}

template <typename T_in, typename T_out>
bool Node<T_in, T_out>::process () {
	const T_out output = activation_fn->process (input);
	if (output != output) return false;
	outputs_buf.insert (output);
	return true;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::mutate (double fitness) {
	activation_fn->mutate (fitness);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::reset (bool resetMemory, bool resetBuffer, bool resetInput) {
	if (resetInput) input = resetValue;
	if (resetMemory) outputs_saved.clear ();
	if (resetBuffer) {
		max_depth_recu = 0;
		outputs_buf = CircularBuffer<T_out> (0);
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
	std::cout << prefix << "Maximum Level of Recurrency in the Network: " << max_depth_recu << std::endl;
	std::cout << prefix << "Current Input Value: " << input << std::endl;
	std::cout << prefix << "Reset Value: " << resetValue << std::endl;
	std::cout << prefix << "Activation Function Parameters: ";
	activation_fn->print (prefix);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::serialize (std::ofstream& outFile) const {
	Serialize (id, outFile);
	Serialize (innovId, outFile);
	Serialize (is_useful, outFile);
	Serialize (max_depth_recu, outFile);
	Serialize (layer, outFile);
	Serialize (index_T_in, outFile);
	Serialize (index_T_out, outFile);
	Serialize (index_activation_fn, outFile);
	activation_fn->serialize (outFile);
	Serialize (input, outFile);
	outputs_buf.serialize (outFile);
	Serialize (outputs_saved, outFile);
	Serialize (resetValue, outFile);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::deserialize (std::ifstream& inFile, ActivationFnBase* activationFn) {
	Deserialize (id, inFile);
	Deserialize (innovId, inFile);
	Deserialize (is_useful, inFile);
	Deserialize (max_depth_recu, inFile);
	Deserialize (layer, inFile);
	Deserialize (index_T_in, inFile);
	Deserialize (index_T_out, inFile);
	Deserialize (index_activation_fn, inFile);
	setActivationFn (activationFn->clone (true));	// clone with parameters doesn't effect anything has we'll overwrite those parameters later
	activation_fn->deserialize (inFile);	// overwrite parameters
	Deserialize (input, inFile);
	outputs_buf.deserialize (inFile);
	Deserialize (outputs_saved, inFile);
	Deserialize (resetValue, inFile);
}

#endif	// NODE_HPP