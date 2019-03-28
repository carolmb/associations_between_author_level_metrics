#include <string>
#include <igraph.h>
#include "parser.h"
#include "util.h"

using string = std::string;

int main() {
    string filename = "data/citation_network.xnet";
    // string filename = "test3.xnet";
    igraph_t *g = xnet2igraph(filename);
    
    return 0;
}