#ifndef INNOVATION_NODE_HPP
#define INNOVATION_NODE_HPP

#include <PNEATM/utils.hpp>
#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>

namespace pneatm {

/**
 * @brief Structure representing the innovation tracker for nodes.
 *
 * The `innovationConn` struct serves as an innovation tracker for nodes in a neural network.
 * It keeps track of the innovation IDs assigned to different nodes based on input and output types,
 * activation functions index and the node's repetition. It also provides a method to retrieve a unique
 * innovation ID.
 */
typedef struct innovationNode {
    /**
     * @brief 4D array that represent the node's innovation ids in the (*input type index*, *output type index*, *activation function index*, *repetition level*) space.
     */
    std::vector<std::vector<std::vector<std::vector<unsigned int>>>> nodeIds;

    /**
     * @brief The next innovation id to give
     */
    unsigned int N_nodeId; 

    /**
     * @brief Constructor of innovationNode
     * 
     */
    innovationNode () :
        nodeIds (1),
        N_nodeId (1)
    {};

    /**
     * @brief Get the innovation ID for a node.
     * @param index_T_in The input type index.
     * @param index_T_out The output type index.
     * @param index_activation_fn The activation function index.
     * @param repetition The occurence of the node.
     * @return The innovation ID for the specified node.
     */
    unsigned int getInnovId (unsigned int index_T_in, unsigned int index_T_out, unsigned int index_activation_fn, unsigned int repetition) {
        while ((unsigned int) nodeIds.size () < index_T_in + 1) {
            nodeIds.push_back ({});
        }
        while ((unsigned int) nodeIds [index_T_in].size () < index_T_out + 1) {
            nodeIds [index_T_in].push_back ({});
        }
        while ((unsigned int) nodeIds [index_T_in][index_T_out].size () < index_activation_fn + 1) {
            nodeIds [index_T_in][index_T_out].push_back ({});
        }
        while ((unsigned int) nodeIds [index_T_in][index_T_out][index_activation_fn].size () < repetition + 1) {
            nodeIds [index_T_in][index_T_out][index_activation_fn].push_back (0);
        }
        if (nodeIds [index_T_in][index_T_out][index_activation_fn][repetition] == 0) {
            nodeIds [index_T_in][index_T_out][index_activation_fn][repetition] = N_nodeId;
            N_nodeId ++;
        }
        return nodeIds [index_T_in][index_T_out][index_activation_fn][repetition];
    }

    /**
     * @brief Print information about the innovation tracker.
     * @param prefix A prefix to print before each line. (default is an empty string)
     */
    void print (const std::string& prefix = "") {
        std::cout << prefix << "Number of attributed innovation: " << N_nodeId - 1 << std::endl;
    }

    /**
     * @brief Serialize the innovationNode instance to an output file stream.
     * @param outFile The output file stream to which the innovationNode instance will be written.
     */
    void serialize (std::ofstream& outFile) {
        Serialize (nodeIds, outFile);
        Serialize (N_nodeId, outFile);
    }

    /**
     * @brief Deserialize a innovationNode instance from an input file stream.
     * @param inFile The input file stream from which the innovationNode instance will be read.
     */
    void deserialize (std::ifstream& inFile) {
        Deserialize (nodeIds, inFile);
        Deserialize (N_nodeId, inFile);
    }

} innovationNode_t;

}

#endif	// INNOVATION_NODE_HPP