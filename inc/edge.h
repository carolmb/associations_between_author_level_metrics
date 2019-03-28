#ifndef __EDGE__
#define __EDGE__

#include <string>
using namespace std;

class Edge
{
public:
	int source,target;
	Edge(int s, int t) :
		source(s), target(t) {}
	~Edge();

};

#endif