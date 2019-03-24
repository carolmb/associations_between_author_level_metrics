#include <string>
#include "graph.h"
#include "parser.h"
#include "util.h"

using string = std::string;

int main() {
    string filename = "../data/citation_network.xnet";
    // string filename = "test3.xnet";
    Graph *g = xnet2graph(filename);
    vector<Type> authors_name = g->get_field_values("authors_name_v");
    // cout << *g;
    return 0;
}