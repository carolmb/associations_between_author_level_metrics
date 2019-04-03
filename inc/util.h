#ifndef __UTIL__
#define __UTIL__

#include <igraph.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <set>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace std;

bool is_alpnum(string);
vector<string> split_str(string &, char);
vector<vector<long long int> > split_vec(vector<string> *, char);

template<class T>
set<T> get_unique_authors(vector<vector<T> > &);

// igraph_vector_t select_ids(igraph_t &g, int begin, int end);
// igraph_t* create_colab(igraph_t *, int, int);
void print_papers_stat(igraph_t&,igraph_t&);
bool comp (int i,int j);
#endif