#ifndef NODE_HPP
#define NODE_HPP

namespace neat {

class Node{
	public:
		Node(int id, int layer);
		Node() {};

	private:
		int id;
		int layer;
		void* sumInput;
		void* sumOutput;
		int in_kind;
		int out_kind;
		int func_id;
};

}

#endif	// NODE_HPP