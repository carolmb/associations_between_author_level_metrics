#include <string>
#include <igraph.h>
#include "parser.h"
#include "util.h"

using string = std::string;

int main() {
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    string filename = "data/citation_net_ge_1990.xnet";
    // string filename = "test3.xnet";
    igraph_t g = xnet2igraph(filename);
    
    return 0;
}