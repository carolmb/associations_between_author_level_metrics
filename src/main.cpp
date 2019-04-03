#include <string>
#include <igraph.h>
#include "parser.h"
#include "util.h"

using string = std::string;

int main() {
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    // string filename = "data/citation_net_ge_1990.xnet";
    string filename = "py/colabs/basic_colab_cut/colab_2010_2014_test_0.8_selected_basic.xnet";
    // string filename = "test3.xnet";
    igraph_t g = xnet2igraph(filename);
    
    return 0;
}
