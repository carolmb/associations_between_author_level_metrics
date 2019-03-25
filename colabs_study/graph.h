#ifndef __GRAPH__
#define __GRAPH__

#include <string>
#include <variant>
#include <vector>
#include "edge.h"
#include "util.h"

using namespace std;

class Graph
{
public:
	int n_vtxs;
	int n_edges;
	bool isdirected;
	vector<Edge> edges;
	vector<string> field_names_str;
	vector<string> field_names_num;
	vector<vector<string> > field_values_str;
	vector<vector<double> > field_values_num;
	Graph(int v, int e, bool d, vector<Edge> eds, vector<string> f_str, 
		vector<vector<string> > fvals_str, vector<string> f_num, 
		vector<vector<double> > fvals_num) : 
		n_vtxs(v), n_edges(e), isdirected(d), edges(eds), 
		field_names_str(f_str), field_values_str(fvals_str), 
		field_names_num(f_num), field_values_num(fvals_num) {}

	void print_field(string field);
	vector<string>* get_field_values_str(string field);
	vector<double>* get_field_values_num(string field);
	string tostring();
};

#endif