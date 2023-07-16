#ifndef INNOVATION_NODE_HPP
#define INNOVATION_NODE_HPP

#include <vector>
#include <iostream>
#include <cstring>

namespace pneatm {

typedef struct innovationNode {
    std::vector<std::vector<std::vector<std::vector<int>>>> nodeIds;
    int N_nodeId = 0;

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
            nodeIds [index_T_in][index_T_out][index_activation_fn].push_back (-1);
        }
        if (nodeIds [index_T_in][index_T_out][index_activation_fn][repetition] == -1) {
            nodeIds [index_T_in][index_T_out][index_activation_fn][repetition] = N_nodeId;
            N_nodeId ++;
        }
        return (unsigned int) nodeIds [index_T_in][index_T_out][index_activation_fn][repetition];
    }

    void print (std::string prefix = "") {
        std::cout << prefix << "Number of attributed innovation: " << N_nodeId << std::endl;
    }
} innovationNode_t;

}

#endif	// INNOVATION_NODE_HPP