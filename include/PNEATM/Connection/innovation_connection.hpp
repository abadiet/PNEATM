#ifndef INNOVATION_CONNECTION_HPP
#define INNOVATION_CONNECTION_HPP

#include <PNEATM/utils.hpp>
#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>

namespace pneatm {

/**
 * @brief Structure representing the innovation tracker for connections.
 *
 * The `innovationConn` struct serves as an innovation tracker for connections in a neural network.
 * It keeps track of the innovation IDs assigned to different connections based on input and output node
 * innovation IDs and the connection's recurrency. It also provides a method to retrieve a unique
 * innovation ID.
 */
typedef struct innovationConn {
    /**
     * @brief 3D array that represent the connection's innovation ids in the (*input node*, *output node*, *connection recurrency level*) space.
     */
    std::vector<std::vector<std::vector<unsigned int>>> connectionIds;

    /**
     * @brief The next innovation id to give.
     */
    unsigned int N_connectionId;

    /**
     * @brief Constructor of innovationConn
     * 
     */
    innovationConn () :
        connectionIds (1),
        N_connectionId (1)
    {};


    /**
     * @brief Get the innovation ID for a connection.
     * @param inNodeInnovId The innovation ID of the input node.
     * @param outNodeInnovId The innovation ID of the output node.
     * @param inNodeRecu The recurrency of the input node.
     * @return The innovation ID for the specified connection.
     */
    unsigned int getInnovId (unsigned int inNodeInnovId, unsigned int outNodeInnovId, unsigned int inNodeRecu) {
        while ((unsigned int) connectionIds.size () < inNodeInnovId + 1) {
            connectionIds.push_back ({});
        }
        while ((unsigned int) connectionIds [inNodeInnovId].size () < outNodeInnovId + 1) {
            connectionIds [inNodeInnovId].push_back ({});
        }
        while ((unsigned int) connectionIds [inNodeInnovId][outNodeInnovId].size () < inNodeRecu + 1) {
            connectionIds [inNodeInnovId][outNodeInnovId].push_back (0);
        }
        if (connectionIds [inNodeInnovId][outNodeInnovId][inNodeRecu] == 0) {
            connectionIds [inNodeInnovId][outNodeInnovId][inNodeRecu] = N_connectionId;
            N_connectionId ++;
        }
        return connectionIds [inNodeInnovId][outNodeInnovId][inNodeRecu];
    }

    /**
     * @brief Print information about the innovation tracker.
     * @param prefix A prefix to print before each line. (default is an empty string)
     */
    void print (const std::string& prefix = "") {
        std::cout << prefix << "Number of attributed innovation: " << N_connectionId - 1 << std::endl;
    }

    /**
     * @brief Serialize the innovationConn instance to an output file stream.
     * @param outFile The output file stream to which the innovationConn instance will be written.
     */
    void serialize (std::ofstream& outFile) {
        Serialize (connectionIds, outFile);
        Serialize (N_connectionId, outFile);
    }

    /**
     * @brief Deserialize a innovationConn instance from an input file stream.
     * @param inFile The input file stream from which the innovationConn instance will be read.
     */
    void deserialize (std::ifstream& inFile) {
        Deserialize (connectionIds, inFile);
        Deserialize (N_connectionId, inFile);
    }
} innovationConn_t;

}

#endif	// INNOVATION_CONNECTION_HPP