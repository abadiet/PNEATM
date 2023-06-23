#ifndef NODE_HPP
#define NODE_HPP

#include <PNEATM/Node/node_base.hpp>
#include <functional>


/* HEADER */

namespace pneatm {

template <typename T_in, typename T_out>
class Node : public NodeBase {
	public:
		Node ();

		void setActivationFn (void* f) override;
		void setActivationFnToIdentity () override;
		void setResetValue (void* value) override;

		void setInput (void* value) override;
		void AddToInput (void* value, float scalar) override;	// TODO: too dirty
		void* getOutput () override;

		void process () override;
		void reset () override;

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
Node<T_in, T_out>::Node () {
	activationFn_isIdentity = false;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setActivationFn (void* f) {
	func = *static_cast<std::function<T_out (T_in)>*> (f);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setActivationFnToIdentity () {
	activationFn_isIdentity = true;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setResetValue (void* value) {
	resetValue = *static_cast<T_in*> (value);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setInput (void* value) {
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
	if (activationFn_isIdentity) {
		output = input;
	} else {
		output = func (input);
	}
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::reset () {
	input = resetValue;
}

#endif	// NODE_HPP