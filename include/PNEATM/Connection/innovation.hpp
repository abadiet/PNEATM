#ifndef INNOVATION_HPP
#define INNOVATION_HPP

#include <vector>

namespace pneatm {

typedef struct innovation {
    std::vector<std::vector<std::vector<int>>> connectionIds;
    int N_connectionId = 0;

    unsigned int getInnovId (unsigned int inNodeId, unsigned int outNodeId, unsigned int inNodeRecu) {
        while ((unsigned int) connectionIds.size () < inNodeId) {
            connectionIds.push_back ({{-1}});
        }
        while ((unsigned int) connectionIds [inNodeId].size () < outNodeId) {
            connectionIds [inNodeId].push_back ({-1});
        }
        while ((unsigned int) connectionIds [inNodeId][outNodeId].size () < inNodeRecu) {
            connectionIds [inNodeId][outNodeId].push_back (-1);
        }
        if (connectionIds [inNodeId][outNodeId][inNodeRecu] == -1) {
            connectionIds [inNodeId][outNodeId][inNodeRecu] = N_connectionId;
            N_connectionId ++;
        }
        return (unsigned int) connectionIds [inNodeId][outNodeId][inNodeRecu];
    }
} innovation_t;

}

#endif	// INNOVATION_HPP