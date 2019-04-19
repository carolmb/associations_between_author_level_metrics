#ifndef __PARSER__
#define __PARSER__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <igraph.h>
#include <ctime>
#include "edge.h"

using namespace std;

int read_vertices(ifstream &, vector<string> &);
int read_edges(ifstream &, vector<Edge> &, vector<double> &, bool &);

void read_field_values(ifstream &, int, string, vector<double> &);
void read_field_values(ifstream &, int, string, vector<string> &);
void add_extra_fields(igraph_t &g, ifstream &input, int n_vtxs, int n_edges);

void add_edges(igraph_t &g, int n_edges, vector<Edge> &p_edges, vector<double> &weights);

void add_numeric_attr_edge(igraph_t &g, vector<double> &attr, string field);
void add_string_attr_edge(igraph_t &g, vector<string> &attr, string field);

void add_numeric_attr_vtx(igraph_t &g, vector<double> &attr, string field);
void add_string_attr_vtx(igraph_t &g, vector<string> &attr, string field);
igraph_t xnet2igraph(string filename);


#endif
