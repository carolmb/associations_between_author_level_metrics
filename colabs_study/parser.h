#ifndef __PARSER__
#define __PARSER__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <variant>
#include <sstream>
#include <iomanip>
#include "edge.h"
#include "graph.h"

using namespace std;

using Type = variant<double, string>;

string read_value(Type value);
std::ostream &operator<<(std::ostream &os, Graph const &g);
int read_vertices(ifstream &input, vector<Type> &names);
int read_edges(ifstream &input, vector<Edge> &edges, bool &isdirected);
void read_field(ifstream &input, int n_elements, string field_type, vector<Type> &field_values);
void read_extra_fields(ifstream &input, int n_vtxs, int n_edges, vector<string> &field_names,
	vector<vector<Type> > &field_values);
Graph* xnet2graph(string filename);

#endif