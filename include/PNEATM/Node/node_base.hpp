#ifndef NODE_BASE_HPP
#define NODE_BASE_HPP

#include <functional>

namespace pneatm {

class NodeBase{
    public:
        virtual ~NodeBase() {};

		virtual void setActivationFn (std::function<void* (void*)> f);
		virtual void setResetValue (void* value);

		virtual void setInput (void* value);
		virtual void AddToInput (void* value, float scalar);	// TODO: too dirty
		virtual void* getOutput ();

		virtual void process ();
		virtual void reset ();

	protected:
		unsigned int id;
		unsigned int layer;
		unsigned int index_T_in;
		unsigned int index_T_out;

	template <typename... Args>
	friend class Genome;
};

}

#endif	// NODE_BASE_HPP