#include <PNEATM/Connection/connection.hpp>

using namespace pneatm;

Connection::Connection (const unsigned int innovId, const unsigned int inNodeId, const unsigned int outNodeId, const unsigned int inNodeRecu, double weight, bool enabled):
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

void Connection::print (std::string prefix) {
	std::cout << prefix << "Innovation ID: " << innovId << std::endl;
	std::cout << prefix << "Input Node ID: " << inNodeId << std::endl;
	std::cout << prefix << "Output Node ID: " << outNodeId << std::endl;
	std::cout << prefix << "Input Node Recurrency: " << inNodeRecu << std::endl;
	std::cout << prefix << "Weight: " << weight << std::endl;
	std::cout << prefix << "Is enabled? " << enabled << std::endl;
	}