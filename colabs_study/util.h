#ifndef __UTIL__
#define __UTIL__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <set>
#include <utility>
#include <map>
#include <algorithm>

#include "graph.h"

using namespace std;

class Graph;

bool is_alpnum(string);
vector<string> split_str(string &, char);
vector<vector<string> > string2vectorofvector(vector<string> *, char);

void get_unique_authors(vector<vector<string> > &,
	vector<vector<string> > &,
	set<pair<string,long long int> > &);

void desambiguation(Graph *);

#endif