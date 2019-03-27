#ifndef __EDGE__
#define __EDGE__

#include <string>
using namespace std;

class Edge
{
public:
	int source,target;
	bool isdirected;
	bool isweighted;
	float weight;
	Edge(int s, int t, bool d, bool w, float w_value) :
		source(s), target(t), isdirected(d), isweighted(w), weight(w_value) {}
	~Edge();

	string tostring();
};

#endif