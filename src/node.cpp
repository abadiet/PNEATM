#include <NEAT/node.hpp>

using namespace neat;

Node::Node(int id, int layer): id(id), layer(layer) {
	sumInput = 0;
	sumOutput = 0;
}
