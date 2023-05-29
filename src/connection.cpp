#include <LRNEAT/connection.hpp>

using namespace neat;

Connection::Connection(int innovId, int inNodeId, int outNodeId, int inNodeRecu, float weight, bool enabled): innovId(innovId), inNodeId(inNodeId), outNodeId(outNodeId), inNodeRecu (inNodeRecu), weight(weight), enabled(enabled) {
}
