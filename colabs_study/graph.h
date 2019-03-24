#ifndef __GRAPH__
#define __GRAPH__

#include <string>
#include <variant>
#include <vector>
#include "edge.h"
#include "util.h"

using namespace std;

using Type = variant<double, string>;

class Graph
{
public:
	int n_vtxs;
	int n_edges;
	bool isdirected;
	vector<Edge> edges;
	vector<string> field_names;
	vector<vector<Type> > field_values;
	Graph(int v, int e, bool d, vector<Edge> eds, vector<string> f, 
		vector<vector<Type> > fvals) : 
		n_vtxs(v), n_edges(e), isdirected(d), edges(eds), 
		field_names(f), field_values(fvals) {}

	void print_field(string field);
	vector<Type>* get_field_values(string field);
};

#endif