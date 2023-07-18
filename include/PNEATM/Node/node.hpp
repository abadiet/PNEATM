#ifndef NODE_HPP
#define NODE_HPP

#include <PNEATM/Node/node_base.hpp>
#include <functional>
#include <iostream>
#include <cstring>
#include <memory>

/* HEADER */

namespace pneatm {

template <typename T_in, typename T_out>
class Node : public NodeBase {
	public:
		Node (bool is_monotyped = false);
		~Node () {};

		void setActivationFn (void* f) override;
		void setResetValue (void* value) override;

		void loadInput (void* value) override;
		void AddToInput (void* value, double scalar) override;	// TODO: too dirty
		void* getOutput () override;

		void process () override;
		void reset () override;

		std::unique_ptr<NodeBase> clone () override;

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