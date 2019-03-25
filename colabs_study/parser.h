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

int read_vertices(ifstream &, vector<string> &);
int read_edges(ifstream &, vector<Edge> &, bool &);

void read_field_values(ifstream &, int, string, vector<double> &);
void read_field_values(ifstream &, int, string, vector<string> &);
void read_extra_fields(ifstream &, int, int, vector<string> &,
	vector<vector<string> > &, vector<string> &, vector<vector<double> > &);
Graph* xnet2graph(string filename);

#endif