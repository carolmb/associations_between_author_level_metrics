#include <iostream>
#include <igraph.h>
#include <fstream>
#include <string>

void read_xnet (std::string filename) {
    std::ifstream input (filename);
    std::string line;
    if (input.is_open()) {
        while (std::getline(input,line)) {
            std::cout << line << std::endl;
        }
        input.close();
    } else {
        std::cout << "deu ruim" << std::endl;
    }
}

int main() {
    std::string filename = "test.xnet";
    read_xnet(filename);
    return 0;
}
