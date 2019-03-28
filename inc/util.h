#ifndef __UTIL__
#define __UTIL__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <set>
#include <unordered_set>
#include <utility>
#include <map>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <igraph.h>

using namespace std;

bool is_alpnum(string);
vector<string> split_str(string &, char);
vector<vector<string> > split_vec(vector<string> *, char);

template<class T>
set<T> get_unique_authors(vector<vector<T> > &);

igraph_t* create_colab(igraph_t *, int, int);
#endif