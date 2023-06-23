#ifndef INNOVATION_HPP
#define INNOVATION_HPP

#include <vector>
#include <iostream>
#include <cstring>

namespace pneatm {

typedef struct innovation {
    std::vector<std::vector<std::vector<int>>> connectionIds;
    int N_connectionId = 0;

    unsigned int getInnovId (unsigned int inNodeId, unsigned int outNodeId, unsigned int inNodeRecu) {
        while ((unsigned int) connectionIds.size () < inNodeId + 1) {
            connectionIds.push_back ({});
        }
        while ((unsigned int) connectionIds [inNodeId].size () < outNodeId + 1) {
            connectionIds [inNodeId].push_back ({});
        }
        while ((unsigned int) connectionIds [inNodeId][outNodeId].size () < inNodeRecu + 1) {
            connectionIds [inNodeId][outNodeId].push_back (-1);
        }
        if (connectionIds [inNodeId][outNodeId][inNodeRecu] == -1) {
            connectionIds [inNodeId][outNodeId][inNodeRecu] = N_connectionId;
            N_connectionId ++;
        }
        return (unsigned int) connectionIds [inNodeId][outNodeId][inNodeRecu];
    }

    void print (std::string prefix = "") {
        std::cout << prefix << "Number of attributed innovation: " << N_connectionId << std::endl;
    }
} innovation_t;

}

#endif	// INNOVATION_HPP