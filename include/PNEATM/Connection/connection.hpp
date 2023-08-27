#ifndef CONNECTION_HPP
#define CONNECTION_HPP

#include <PNEATM/utils.hpp>
#include <iostream>

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
		 * @param id The ID of the connection.
		 * @param innovId The innovation ID of the connection.
		 * @param inNodeId The ID of the input node.
		 * @param outNodeId The ID of the output node.
		 * @param inNodeRecu The recurrency of the input node: 0 means no recurrency.
		 * @param weight The weight of the connection.
		 * @param enabled Set to true if the connection is enabled, false if it is disabled.
		 */
		Connection (const unsigned int id, const unsigned int innovId, const unsigned int inNodeId, const unsigned int outNodeId, const unsigned int inNodeRecu, double weight, bool enabled);

		/**
		 * @brief Constructor for the Connection class from an input file stream.
		 * @param inFile The input file stream.
		 */
		Connection (std::ifstream& inFile);

		/**
		 * @brief Constructor for the Connection class
		 *
		 * This constuctor should not be used. It is implemented to avoid warnings with calls to undordered_map<pneatm::Connection>::operator[].
		 */
		Connection () {std::cout << "[ERROR] The constructor pneatm::Connection::Connection() might not be used." << std::endl;};

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
		void print (const std::string& prefix = "") const;

		/**
		 * @brief Serialize the Connection instance to an output file stream.
		 * @param outFile The output file stream to which the Connection instance will be written.
		 */
		void serialize (std::ofstream& outFile) const;

		/**
		 * @brief Deserialize a Connection instance from an input file stream.
		 * @param inFile The input file stream from which the Connection instance will be read.
		 */
		void deserialize (std::ifstream& inFile);

	private:
		unsigned int id;
		unsigned int innovId;
		unsigned int inNodeId;
		unsigned int outNodeId;
		unsigned int inNodeRecu;
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