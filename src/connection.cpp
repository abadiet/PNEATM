#include <PNEATM/Connection/connection.hpp>

using namespace pneatm;

Connection::Connection (const unsigned int id, const unsigned int innovId, const unsigned int inNodeId, const unsigned int outNodeId, const unsigned int inNodeRecu, double weight, bool enabled):
    id (id),
    innovId (innovId),
    inNodeId (inNodeId),
    outNodeId (outNodeId),
    inNodeRecu (inNodeRecu),
    weight (weight),
    enabled (enabled) {
}

Connection::Connection (std::ifstream& inFile) {
    deserialize (inFile);
}

Connection& Connection::operator=(const Connection& other) {
    if (this == &other) {
        return *this;  // Self-assignment check
    }

    weight = other.weight;
    enabled = other.enabled;

    return *this;
}

void Connection::print (const std::string& prefix) const {
	std::cout << prefix << "ID: " << id << std::endl;
	std::cout << prefix << "Innovation ID: " << innovId << std::endl;
	std::cout << prefix << "Input Node ID: " << inNodeId << std::endl;
	std::cout << prefix << "Output Node ID: " << outNodeId << std::endl;
	std::cout << prefix << "Input Node Recurrency: " << inNodeRecu << std::endl;
	std::cout << prefix << "Weight: " << weight << std::endl;
	std::cout << prefix << "Is enabled? " << enabled << std::endl;
}

void Connection::serialize (std::ofstream& outFile) const {
    Serialize (id, outFile);
    Serialize (innovId, outFile);
    Serialize (inNodeId, outFile);
    Serialize (outNodeId, outFile);
    Serialize (inNodeRecu, outFile);
    Serialize (weight, outFile);
    Serialize (enabled, outFile);
}

void Connection::deserialize (std::ifstream& inFile) {
    Deserialize (id, inFile);
    Deserialize (innovId, inFile);
    Deserialize (inNodeId, inFile);
    Deserialize (outNodeId, inFile);
    Deserialize (inNodeRecu, inFile);
    Deserialize (weight, inFile);
    Deserialize (enabled, inFile);
}
