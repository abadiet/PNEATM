#ifndef NODE_HPP
#define NODE_HPP

#include <PNEATM/Node/node_base.hpp>
#include <functional>

namespace pneatm {

template <typename T_in, typename T_out>
class Node : public NodeBase {
	public:
		Node (unsigned int ID, unsigned int lay, unsigned int iT_in, unsigned int iT_out, std::function<T_out (T_in)> func, T_in resetValue);
		Node () {};

		void setActivationFn (std::function<void* (void*)> f) override;
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

#endif	// NODE_HPP