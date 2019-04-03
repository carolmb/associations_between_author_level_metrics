#include <string>
#include <igraph.h>
#include "parser.h"
#include "util.h"

using string = std::string;

int main() {
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    string filename = "data/citation_net_ge_1990.xnet";
    igraph_t citation_net = xnet2igraph(filename);
    filename = "colab_1990_1994_test_0.8_selected_basic.xnet";
    igraph_t g = xnet2igraph(filename);
    print_papers_stat(g,citation_net);
    return 0;
}