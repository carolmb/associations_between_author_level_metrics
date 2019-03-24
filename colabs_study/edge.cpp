#include "edge.h"

Edge::~Edge() {}

string Edge::tostring() {
	string output = "(" + to_string(source) + ", " + to_string(target);
	if (isweighted)
		output += ", " + to_string(weight);
	output += ")";
	return output;
}