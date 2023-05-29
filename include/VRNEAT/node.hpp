#ifndef NODE_HPP
#define NODE_HPP

namespace neat {

class Node{
	public:
		Node (int id, int layer, int in_kind, int out_kind, int func_id);
		Node () {};

	protected:
		int layer;
		void* input;
		void* output;
		int func_id;

	private:
		int id;
		int in_kind;
		int out_kind;
};

}

#endif	// NODE_HPP