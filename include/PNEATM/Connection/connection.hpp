#ifndef CONNECTION_HPP
#define CONNECTION_HPP

#include <PNEATM/utils.hpp>
#include <iostream>
#include <cstring>
#include <fstream>

namespace pneatm {

/**
 * @brief Class representing a connection between nodes in a neural network.
 *
 * The `Connection` class represents a connection between two nodes in a neural network. It holds
 * information about the connection's innovation ID, input node ID, output node ID, whether it is
 * recurrent, weight, and whether the connection is enabled or disabled.
 */
class Connection {
	public:
		/**
		 * @brief Constructor for the Connection class.
		 * @param innovId The innovation ID of the connection.
		 * @param inNodeId The ID of the input node.
		 * @param outNodeId The ID of the output node.
		 * @param inNodeRecu The recurrency of the input node: 0 means no recurrency.
		 * @param weight The weight of the connection.
		 * @param enabled Set to true if the connection is enabled, false if it is disabled.
		 */
		Connection (const unsigned int innovId, const unsigned int inNodeId, const unsigned int outNodeId, const unsigned int inNodeRecu, double weight, bool enabled);

		/**
		 * @brief Assignment operator for the Connection class.
		 * @param other The Connection object to assign from.
		 * @return A reference to the assigned Connection object.
		 */
		Connection& operator= (const Connection& other);

		/**
		 * @brief Print information about the Connection.
		 * @param prefix A prefix to print before each line. (default is an empty string)
		 */
		void print (const std::string& prefix = "");

		void serialize (std::ofstream& outFile);

	private:
		const unsigned int innovId;
		const unsigned int inNodeId;
		const unsigned int outNodeId;
		const unsigned int inNodeRecu;
		double weight;
		bool enabled;

	template <typename... Args>
	friend class Genome;
	template <typename... Args>
	friend class Population;
	template <typename... Args>
	friend class Species;
};

}

#endif	// CONNECTION_HPP