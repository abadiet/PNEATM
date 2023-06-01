#include <VRNEAT/node.hpp>

using namespace vrneat;

Node::Node (int id, int layer, int in_kind, int out_kind): id(id), layer(layer), in_kind (in_kind), out_kind (out_kind) {
	sumInput = 0;
	sumOutput = 0;
	func_id = -1;
}
