#ifndef NODE_HPP
#define NODE_HPP

#include <PNEATM/Node/node_base.hpp>
#include <functional>
#include <iostream>
#include <cstring>


/* HEADER */

namespace pneatm {

template <typename T_in, typename T_out>
class Node : public NodeBase {
	public:
		Node ();
		~Node () {};

		void setActivationFn (void* f) override;
		void setResetValue (void* value) override;

		void loadInput (void* value) override;
		void AddToInput (void* value, float scalar) override;	// TODO: too dirty
		void* getOutput () override;

		void process () override;
		void reset () override;

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

#include <PNEATM/utils.hpp>

template <typename T_in, typename T_out>
Node<T_in, T_out>::Node () {
	func = [] (T_in input) {	// default activation function is the identity (useful for bias/inputs/outputs)
		return input;
	};
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
void Node<T_in, T_out>::AddToInput (void* value, float scalar) {
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
void Node<T_in, T_out>::print (std::string prefix) {
	std::cout << prefix << "ID: " << id << std::endl;
	std::cout << prefix << "Layer: " << layer << std::endl;
	std::cout << prefix << "Input Type ID: " << index_T_in << std::endl;
	std::cout << prefix << "Output Type ID: " << index_T_out << std::endl;
	std::cout << prefix << "Current Input Value: " << input << std::endl;
	std::cout << prefix << "Current Output Value: " << output << std::endl;
	std::cout << prefix << "Reset Value: " << resetValue << std::endl;
}

#endif	// NODE_HPP