#ifndef __UTIL__
#define __UTIL__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <variant>
#include <sstream>
#include <iomanip>

#include "graph.h"

using namespace std;

using Type = variant<double, string>;

class Graph;

string get_Type_string(Type value);
std::ostream &operator<<(std::ostream &os, Graph const &g);

vector<string> string2vectorofstring(string &str);
vector<vector<string> > string2vectorofvector(vector<string> &vec);
#endif