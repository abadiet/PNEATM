#ifndef CONNECTION_HPP
#define CONNECTION_HPP

namespace vrneat {

class Connection{
	public:
		Connection(int innovId, int inNodeId, int outNodeId, int inNodeRecu, float weight, bool enabled);
		Connection() {};

	private:
		int innovId;
		int inNodeId;
		int outNodeId;
		int inNodeRecu;
		float weight;
		bool enabled;
};

}

#endif	// CONNECTION_HPP