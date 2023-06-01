#ifndef NODE_HPP
#define NODE_HPP

namespace vrneat {

class Node{
	public:
		Node (int id, int layer, int in_kind, int out_kind, int func_id);
		Node () {};

	private:
		int id;
		int in_kind;
		int out_kind;

		int layer;
		void* input;
		void* output;
		int func_id;
};

}

#endif	// NODE_HPP