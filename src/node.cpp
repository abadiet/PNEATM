#include <VRNEAT/Node/node.hpp>

using namespace vrneat;

template <typename T_in, typename T_out>
Node<T_in, T_out>::Node (unsigned int ID, unsigned int lay, unsigned int iT_in, unsigned int iT_out, std::function<T_out (T_in)> func, T_in resetValue):
	func (func),
	resetValue (resetValue)
{
	id = ID;
	layer = lay;
	index_T_in = iT_in;
	index_T_out = iT_out;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setActivationFn (std::function<void* (void*)> f) {
	func = [=] (T_in input) {return (T_out) f ((void*) input);};
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setResetValue (void* value) {
	resetValue = (T_in) value;
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::setInput (void* value) {
	setInput ((T_in) value);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::AddToInput (void* value, float scalar) {
	input += (T_in) value * scalar;
}


template <typename T_in, typename T_out>
void* Node<T_in, T_out>::getOutput () {
	return (void*) output; 
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::process () {
	output = func (input);
}

template <typename T_in, typename T_out>
void Node<T_in, T_out>::reset () {
	input = resetValue;
}