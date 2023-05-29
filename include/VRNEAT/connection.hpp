#ifndef CONNECTION_HPP
#define CONNECTION_HPP

namespace neat {

class Connection{
	public:
		Connection(int innovId, int inNodeId, int outNodeId, int inNodeRecu, float weight, bool enabled);
		Connection() {};

	protected:
		int innovId;
		int inNodeId;
		int outNodeId;
		int inNodeRecu;
		float weight;
		bool enabled;
};

}

#endif	// CONNECTION_HPP