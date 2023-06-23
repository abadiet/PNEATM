#ifndef CONNECTION_HPP
#define CONNECTION_HPP

#include <iostream>
#include <cstring>

namespace pneatm {

class Connection {
	public:
		Connection (const unsigned int innovId, const unsigned int inNodeId, const unsigned int outNodeId, const unsigned int inNodeRecu, float weight, bool enabled);

		Connection& operator= (const Connection& other);

		void print (std::string prefix = "");

	private:
		const unsigned int innovId;
		const unsigned int inNodeId;
		const unsigned int outNodeId;
		const unsigned int inNodeRecu;
		float weight;
		bool enabled;

	template <typename... Args>
	friend class Genome;

	template <typename... Args>
	friend class Population;
};

}

#endif	// CONNECTION_HPP