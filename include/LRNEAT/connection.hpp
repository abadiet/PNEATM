#ifndef CONNECTION_HPP
#define CONNECTION_HPP

namespace neat {

class Connection{
	public:
		int innovId;
		int inNodeId;
		int outNodeId;
		float weight;
		bool enabled;
		bool isRecurrent;

		Connection(int innovId, int inNodeId, int outNodeId, float weight, bool enabled, bool isRecurrent);
		Connection() {};
};

}

#endif	// CONNECTION_HPP