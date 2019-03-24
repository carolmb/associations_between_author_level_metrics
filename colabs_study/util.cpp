#include "util.h"

string get_Type_string(Type value) {
	try {
    	double temp = get<double>(value); // w contains int, not float: will throw
		return to_string(temp);
	} catch (const std::bad_variant_access&) {}

	try {
    	string temp = get<string>(value); // w contains int, not float: will throw
		return temp;
	} catch (const std::bad_variant_access&) {}

}

std::ostream &operator<<(std::ostream &os, Graph const &g) { 
	string output = "Number of vertices: " + to_string(g.n_vtxs) + " Number of edges: " + to_string(g.n_edges) + "\n";
	if (g.n_edges)
 		output += "Edge list:\n";
	for (int i = 0; i < g.n_edges; i++) {
		if (i < 10 || i > g.n_edges - 10) {
			Edge e = g.edges[i];
			output += e.tostring() + ", ";	
		}
	}

	int n_fields = g.field_names.size();
	if (n_fields) {
		output += "Fields and values:\n";
	}


	for (int i = 0; i < n_fields; i++) {
		string field_name = g.field_names[i];
		output += field_name + ": ";
		vector<Type> field_values = g.field_values[i];
		int n_values = field_values.size();
		
		for (int i = 0; i < n_values-1; i++) {
			output += get_Type_string(field_values[i]) + ", ";
		}
		output += get_Type_string(field_values[n_values-1]) + "\n";
	}
    
    return os << output;
}

vector<string> string2vectorofstring(string &str) {
	istringstream ss(str);
	string token;

	vector<string> output;
	while(getline(ss, token, ',')) {
		output.push_back(token);
	}
	return output;
}

vector<vector<string> > string2vectorofvector(vector<string> &vec) {
	vector<vector<string> > output;
	for (vector<string>::iterator it = vec.begin(); it != vec.end(); it++) {
		vector<string> current = string2vectorofstring(*it);
		output.push_back(current);
	}
	return output;
}