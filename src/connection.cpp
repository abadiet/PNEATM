#include <PNEATM/Connection/connection.hpp>

using namespace pneatm;

Connection::Connection (const unsigned int innovId, const unsigned int inNodeId, const unsigned int outNodeId, const unsigned int inNodeRecu, float weight, bool enabled):
    innovId (innovId),
    inNodeId (inNodeId),
    outNodeId (outNodeId),
    inNodeRecu (inNodeRecu),
    weight (weight),
    enabled (enabled) {
}

Connection& Connection::operator=(const Connection& other) {
    if (this == &other) {
        return *this;  // Self-assignment check
    }

    weight = other.weight;
    enabled = other.enabled;

    return *this;
}
