#include <string>
#include <igraph.h>
#include "parser.h"
#include "util.h"

using string = std::string;

int main() {
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    string filename = "data/citation_net_ge_1990.xnet";
    // string filename = "test3.xnet";
    igraph_t citation_net = xnet2igraph(filename);

    int delta = 4;
    for (int i = 1990; i <= 2010; i++) {
        cout << "Current interval " << i << " until " << i+delta << endl;
        filename = "py/colabs/original/colab_"+to_string(i)+"_"+to_string(i+delta)+"_test";
        cout << filename << endl;
        igraph_t g = xnet2igraph(filename);

        igraph_t giant = get_giant_component(g);
        igraph_destroy(&g);
    }
    return 0;
}
