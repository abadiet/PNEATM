#ifndef INNOVATION_CONNECTION_HPP
#define INNOVATION_CONNECTION_HPP

#include <vector>
#include <iostream>
#include <cstring>

namespace pneatm {

typedef struct innovationConn {
    std::vector<std::vector<std::vector<int>>> connectionIds;
    int N_connectionId = 0;

    unsigned int getInnovId (unsigned int inNodeInnovId, unsigned int outNodeInnovId, unsigned int inNodeRecu) {
        while ((unsigned int) connectionIds.size () < inNodeInnovId + 1) {
            connectionIds.push_back ({});
        }
        while ((unsigned int) connectionIds [inNodeInnovId].size () < outNodeInnovId + 1) {
            connectionIds [inNodeInnovId].push_back ({});
        }
        while ((unsigned int) connectionIds [inNodeInnovId][outNodeInnovId].size () < inNodeRecu + 1) {
            connectionIds [inNodeInnovId][outNodeInnovId].push_back (-1);
        }
        if (connectionIds [inNodeInnovId][outNodeInnovId][inNodeRecu] == -1) {
            connectionIds [inNodeInnovId][outNodeInnovId][inNodeRecu] = N_connectionId;
            N_connectionId ++;
        }
        return (unsigned int) connectionIds [inNodeInnovId][outNodeInnovId][inNodeRecu];
    }

    void print (std::string prefix = "") {
        std::cout << prefix << "Number of attributed innovation: " << N_connectionId << std::endl;
    }
} innovationConn_t;

}

#endif	// INNOVATION_CONNECTION_HPP